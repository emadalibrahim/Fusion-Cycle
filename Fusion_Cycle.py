import pandas as pd
import numpy as np
import os
import torch
import sys
import copy
from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence
from lightning import pytorch as pl
from pathlib import Path
from chemprop import data, featurizers,models
from chemprop.models import multi
from chemprop.data.collate import BatchMolGraph, collate_batch
from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.data.datasets import Datum, MoleculeDataset, MulticomponentDataset, ReactionDataset
from chemprop.models import MulticomponentMPNN
from chemprop.nn.agg import Aggregation, MeanAggregation
from chemprop.nn.hparams import HasHParams
from chemprop.nn.message_passing import MessagePassing
from chemprop.nn.metrics import ChempropMetric
from chemprop.nn.predictors import Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.nn.utils import Activation, get_activation_function
from chemprop.conf import DEFAULT_HIDDEN_DIM
from torch.utils.data import DataLoader, Dataset
from torch import Tensor, nn
from Density.QSPR import fill_solvent_density

BASE_DIR = Path(__file__).resolve().parent

# Detect number of available CUDA devices
num_devices = torch.cuda.device_count()

if num_devices > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    print(f"Using GPU device 0 (total available: {num_devices})")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")
    print("No GPU detected, running on CPU")


class ComponentMolGraph(NamedTuple):
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
    w_fp: float = 1.0


class ComponentDatum(NamedTuple):
    mg: ComponentMolGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


@dataclass(repr=False, eq=False, slots=True)
class BatchComponentMolGraph(BatchMolGraph):
    mgs: InitVar[Sequence[ComponentMolGraph]]
    w_fps: Tensor = field(init=False)
    __is_empty: bool = field(init=False)

    def __post_init__(self, mgs):
        self._BatchMolGraph__size = len(mgs)
        self.__is_empty = True
        Vs, Es, edge_indexes, rev_edge_indexes, batch_indexes, w_fps = [], [], [], [], [], []
        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            if mg is None:
                continue
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))
            w_fps.append(mg.w_fp)
            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float() if Vs else None
        self.E = torch.from_numpy(np.concatenate(Es)).float() if Es else None
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long() if edge_indexes else None
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long() if rev_edge_indexes else None
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long() if batch_indexes else None
        self.w_fps = torch.from_numpy(np.array(w_fps)).float() if w_fps else None
        if Vs:
            self.__is_empty = False

    def to(self, device: str | torch.device):
        if not self.is_empty():
            super(BatchComponentMolGraph, self).to(device)
            self.w_fps = self.w_fps.to(device)

    def is_empty(self) -> bool:
        return self.__is_empty


class BatchComponentDatum(NamedTuple):
    bmg: BatchComponentMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


class MixtureBatch(NamedTuple):
    bmgs: list[BatchMolGraph | BatchComponentMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_component(batch: Iterable[Datum]) -> BatchComponentDatum:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)
    return BatchComponentDatum(
        BatchComponentMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


def collate_mixture(batches: Iterable[Iterable[ComponentDatum | Datum]]) -> MixtureBatch:
    tbs = [
        collate_batch(batch) if isinstance(batch[0], Datum) else collate_component(batch)
        for batch in zip(*batches)
    ]
    return MixtureBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )


@dataclass
class ComponentDatapoint(MoleculeDatapoint):
    w_fp: np.ndarray | None = None


@dataclass
class ComponentDataset(MoleculeDataset, Dataset[ComponentMolGraph]):
    data: list[ComponentDatapoint]

    @property
    def w_fps(self) -> np.ndarray:
        return np.array([d.w_fp for d in self.data])

    def __getitem__(self, idx: int) -> ComponentDatum:
        d = self.data[idx]
        mg = ComponentMolGraph(w_fp=d.w_fp, *self.mg_cache[idx]) if d.mol else None
        return ComponentDatum(
            mg, self.V_ds[idx], self.X_d[idx], self.Y[idx], d.weight, d.lt_mask, d.gt_mask
        )


@dataclass(repr=False, eq=False)
class MixtureDataset(MulticomponentDataset):
    datasets: list[MoleculeDataset | ReactionDataset | ComponentDataset]

    def __getitem__(self, idx: int) -> list[ComponentDatum | Datum]:
        return [dset[idx] for dset in self.datasets]


class MulticomponentMessagePassing(nn.Module, HasHParams):
    def __init__(self, blocks: Sequence[MessagePassing], groups: Sequence[Sequence[int]], shared: bool = False):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "blocks": [block.hparams for block in blocks],
            "groups": groups,
            "shared": shared,
        }
        if len(blocks) == 0:
            raise ValueError("arg 'blocks' was empty!")
        if groups is None:
            raise ValueError("arg 'groups' was empty!")
        if not shared and len(blocks) != len(groups):
            raise ValueError("len(groups) must equal len(blocks) when shared is False")

        self.groups = groups
        self.shared = shared
        self.blocks = nn.ModuleList()
        if shared:
            self.blocks.extend([blocks[0]] * len(groups))
        else:
            for g_idx, group in enumerate(groups):
                self.blocks.extend([blocks[g_idx]] * len(group))

    @property
    def output_dim(self) -> int:
        return sum(block.output_dim for block in self.blocks)

    def forward(self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor | None]) -> list[Tensor]:
        if V_ds is None:
            return [block(bmg) for block, bmg in zip(self.blocks, bmgs)]
        return [block(bmg, V_d) for block, bmg, V_d in zip(self.blocks, bmgs, V_ds)]


class MixtureMessagePassing(nn.Module):
    def __init__(
        self,
        d_v: int = DEFAULT_HIDDEN_DIM,
        d_e: int | None = None,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
        depth: int = 1,
        activation: str | Activation = Activation.RELU,
    ):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "d_v": d_v,
            "d_e": d_e,
            "d_h": d_h,
            "d_vd": d_vd,
            "bias": bias,
            "depth": depth,
            "activation": activation,
        }
        self.depth = depth
        self.tau = get_activation_function(activation)
        self.W_i = nn.Linear(d_v, d_h, bias)
        self.W_h = nn.Linear(d_h, d_h, bias)
        self.W_o = None
        self.W_d = None

    def forward(self, V: list[Tensor]):
        H_0 = self.W_i(torch.stack(V))
        H = self.tau(H_0)
        for _ in range(self.depth):
            H_t = torch.transpose(H, 0, 1)
            M_t = H_t.unsqueeze(2).expand(-1, -1, H_t.size(1), -1)
            M_t = self.W_h(M_t)
            mask = ~torch.eye(H_t.size(1), dtype=bool, device=H_t.device).unsqueeze(0)
            M_t = (M_t * mask.unsqueeze(-1)).sum(dim=1)
            M_t = torch.transpose(M_t, 0, 1)
            H = self.tau(H_0 + M_t)
        return [h for h in H]


class MixtureAggregation(nn.Module, HasHParams):
    output_dim: int

    def __init__(
        self,
        graph_agg: Aggregation,
        groups: Sequence[Sequence[int]],
        fp_dims: Sequence[int],
        mixmp: MixtureMessagePassing | None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "graph_agg": graph_agg.hparams,
            "groups": groups,
            "fp_dims": fp_dims,
            "mixmp": None if mixmp is None else mixmp.hparams,
        }
        self.graph_agg = graph_agg
        self.groups = groups
        self.fp_dims = fp_dims
        self.mixmp = mixmp

    @abstractmethod
    def forward(self, Hs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]) -> Tensor:
        ...

    def mol_forward(self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]):
        Hs, w_fps, Hs_batch = zip(
            *[
                (
                    self.graph_agg(H_v, torch.unique(bmg.batch, return_inverse=True)[1]),
                    bmg.w_fps if isinstance(bmg, BatchComponentMolGraph) else None,
                    torch.unique(bmg.batch),
                )
                if bmg.batch is not None
                else (None, None, None)
                for H_v, bmg in zip(H_vs, bmgs)
            ]
        )
        return list(Hs), list(w_fps), list(Hs_batch)

    def complete_sparse_batch(self, Hs: list[Tensor], w_fps: list[Tensor], Hs_batch: list[Tensor]):
        Hs = self._complete_sparse_tensorlist(Hs, Hs_batch, self.fp_dims)
        if not all(f is None for f in w_fps):
            w_fps = self._complete_sparse_tensorlist(w_fps, Hs_batch, [None for _ in range(len(Hs_batch))])
        return Hs, w_fps, Hs_batch

    def _complete_sparse_tensorlist(self, Hs: list[Tensor], Hs_batch: list[Tensor], dim: list[int]):
        batch_size = max(H_b.max().item() for H_b in Hs_batch if H_b is not None)
        device_ = [H.device for H in Hs if H is not None][0]
        compl_Hs = []
        for n_idx, (n_H, n_H_batch) in enumerate(zip(Hs, Hs_batch)):
            shape = (batch_size + 1, dim[n_idx]) if dim[n_idx] else (batch_size + 1,)
            compl_H = torch.zeros(shape, dtype=torch.float32, device=device_)
            if n_H is not None and n_H_batch is not None:
                compl_H[n_H_batch] = n_H
            compl_Hs.append(compl_H)
        return compl_Hs


class WeightedSumAggregation(MixtureAggregation):
    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]) -> Tensor:
        Hs, w_fps, Hs_batch = self.mol_forward(H_vs, bmgs)
        Hs, w_fps, Hs_batch = self.complete_sparse_batch(Hs, w_fps, Hs_batch)
        if self.mixmp is not None:
            Hs = self.mixmp(Hs)

        combined_Hs = []
        for group in self.groups:
            if len(group) == 1:
                combined_Hs.append(Hs[group[0]])
                continue
            group_Hs = torch.stack([Hs[idx] for idx in group])
            group_w_fps = torch.stack([w_fps[idx] for idx in group])
            combined_Hs.append(torch.einsum("nb,nbd->bd", group_w_fps, group_Hs))
        return torch.cat(combined_Hs, 1)


class MixtureMPNN(MulticomponentMPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        mix_mpn: MixtureMessagePassing | None = None,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
            X_d_transform,
        )
        self.agg: MixtureAggregation

    def fingerprint(
        self,
        bmgs: Iterable[BatchComponentMolGraph | BatchMolGraph],
        V_ds: Iterable[Tensor],
        X_d: Tensor | None = None,
    ) -> Tensor:
        H_vs = self.message_passing(bmgs, V_ds)
        H = self.agg(H_vs, bmgs)
        H = self.bn(H)
        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location=map_location, weights_only=False)
        try:
            hparams = copy.deepcopy(d["hyper_parameters"])
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        hparams["message_passing"]["blocks"] = [
            block_hparams.pop("cls")(**block_hparams)
            for block_hparams in hparams["message_passing"]["blocks"]
        ]

        graph_agg_hparams = hparams["agg"].get("graph_agg")
        if graph_agg_hparams is None:
            hparams["agg"]["graph_agg"] = MeanAggregation()
        else:
            hparams["agg"]["graph_agg"] = graph_agg_hparams.pop("cls")(**graph_agg_hparams)

        mixmp_hparams = hparams["agg"].get("mixmp")
        if mixmp_hparams is None:
            fp_dim = hparams["agg"]["fp_dims"][0]
            hparams["agg"]["mixmp"] = MixtureMessagePassing(d_v=fp_dim, d_h=fp_dim, depth=1)
        else:
            hparams["agg"]["mixmp"] = mixmp_hparams.pop("cls")(**mixmp_hparams)

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        if not hasattr(submodules["predictor"].criterion, "_defaults"):
            submodules["predictor"].criterion = submodules["predictor"].criterion.__class__(
                task_weights=submodules["predictor"].criterion.task_weights
            )

        return submodules, state_dict, hparams


for _name in (
    "MulticomponentMessagePassing",
    "MixtureMessagePassing",
    "WeightedSumAggregation",
    "MixtureMPNN",
):
    setattr(sys.modules["__main__"], _name, globals()[_name])


class model:
    def __init__(self, N_iteration=10, thresh=0.99, Num_ensembles = 5, mixture = False, segment = False):
        self.N_iteration = N_iteration
        self.thresh = thresh
        self.Num_ensembles = Num_ensembles
        self.R = 1.98720425864083/1000 # Kcal K-1 mol-1
        self.mixture = mixture
        self.segment = "Segment" if segment else "Full"

    def _model_path(self, *parts):
        return str(BASE_DIR.joinpath("trained_models", *parts))

    def _mixture_model_path(self, filename):
        return self._model_path("Mixture_models", self.segment, filename)

    def _prepare_mixture_dataframe(self, df_test, x):
        df_mix = df_test.copy()
        if "solute_smiles_canonical" not in df_mix and "mol_solute" in df_mix:
            df_mix["solute_smiles_canonical"] = df_mix["mol_solute"]
        if "solvent1_smiles_canonical" not in df_mix:
            if "mol_solvent1" in df_mix:
                df_mix["solvent1_smiles_canonical"] = df_mix["mol_solvent1"]
            elif "solvent_smiles_canonical" in df_mix:
                df_mix["solvent1_smiles_canonical"] = df_mix["solvent_smiles_canonical"]
        if "solvent2_smiles_canonical" not in df_mix:
            if "mol_solvent2" in df_mix:
                df_mix["solvent2_smiles_canonical"] = df_mix["mol_solvent2"]
            else:
                df_mix["solvent2_smiles_canonical"] = None
        if "molefrac" not in df_mix:
            if "frac_solvent1" in df_mix:
                df_mix["molefrac"] = df_mix["frac_solvent1"]
            else:
                df_mix["molefrac"] = 1.0

        required = [
            "solute_smiles_canonical",
            "solvent1_smiles_canonical",
            "solvent2_smiles_canonical",
            "molefrac",
            "Temperature [K]",
        ]
        missing = [column for column in required if column not in df_mix]
        if missing:
            raise ValueError(f"Missing required mixture columns: {missing}")

        x_array = np.asarray(x).reshape(-1)
        if len(x_array) != len(df_mix):
            raise ValueError(f"Expected {len(df_mix)} solute mole fractions, got {len(x_array)}")
        df_mix["x"] = x_array
        df_mix["molefrac"] = df_mix["molefrac"].fillna(1.0)
        if "MP_std" not in df_mix:
            df_mix["MP_std"] = 0.0
        return df_mix

    def _prepare_density_dataframe(self, df):
        df_density = df.copy()

        if not self.mixture:
            return df_density

        if "solvent1_smiles_canonical" not in df_density and "mol_solvent1" in df_density:
            df_density["solvent1_smiles_canonical"] = df_density["mol_solvent1"]
        if "solvent2_smiles_canonical" not in df_density:
            if "mol_solvent2" in df_density:
                df_density["solvent2_smiles_canonical"] = df_density["mol_solvent2"]
            else:
                df_density["solvent2_smiles_canonical"] = None
        if "molefrac" not in df_density:
            if "frac_solvent1" in df_density:
                df_density["molefrac"] = df_density["frac_solvent1"]
            else:
                df_density["molefrac"] = 1.0

        if "solvent_avg_density" not in df_density:
            df_density["solvent_avg_density"] = np.nan

        avg_missing = df_density["solvent_avg_density"].isna()
        if avg_missing.any():
            df_density = self._fill_component_density(
                df_density, "solvent1_smiles_canonical", "solvent1_density"
            )
            df_density = self._fill_component_density(
                df_density, "solvent2_smiles_canonical", "solvent2_density"
            )
            solvent2_missing = df_density["solvent2_smiles_canonical"].apply(self._missing_smiles)
            df_density.loc[solvent2_missing, "solvent2_density"] = df_density.loc[
                solvent2_missing, "solvent1_density"
            ]

            x1 = df_density["molefrac"].fillna(1.0).astype(float)
            x2 = 1.0 - x1
            denominator = (x1 / df_density["solvent1_density"]) + (x2 / df_density["solvent2_density"])
            calculated_avg_density = 1.0 / denominator
            df_density.loc[avg_missing, "solvent_avg_density"] = calculated_avg_density[avg_missing]

        df_density["solvent_density"] = df_density["solvent_avg_density"]
        if "solvent_smiles_canonical" not in df_density:
            df_density["solvent_smiles_canonical"] = df_density["solvent1_smiles_canonical"]
        return df_density

    def _fill_component_density(self, df, smiles_column, density_column):
        if smiles_column not in df:
            raise ValueError(f"Missing required mixture column: {smiles_column}")
        if density_column not in df:
            df[density_column] = np.nan

        missing = df[density_column].isna() & ~df[smiles_column].apply(self._missing_smiles)
        if not missing.any():
            return df

        density_input = pd.DataFrame(
            {
                "solvent_smiles_canonical": df.loc[missing, smiles_column],
                "solvent_density": df.loc[missing, density_column],
            },
            index=df.index[missing],
        )
        density_output = fill_solvent_density(density_input)
        df.loc[missing, density_column] = density_output["solvent_density"]
        return df

    def _missing_smiles(self, smi):
        return smi is None or pd.isna(smi) or smi == ""

    # Function that takes in a list of single molecule
    # model paths and a test dataframe
    # It returns a list of predictions for the smiles in
    # the 'solute_smiles_canonical' columns of the dataframe
    def predict_single(self,checkpoint_path_list,df_test):
        pred_list = []
        for checkpoint_path in checkpoint_path_list:
            if '.ckpt' in str(checkpoint_path):
                mpnn = models.MPNN.load_from_checkpoint(checkpoint_path,map_location=device)
            else:
                mpnn = models.MPNN.load_from_file(checkpoint_path,map_location=device)
            smiles_columns = 'solute_smiles_canonical' # name of the column containing SMILES strings
            smis = df_test[smiles_columns].values
            test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
            featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
            test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
            test_loader = data.build_dataloader(test_dset, shuffle=False,batch_size=1)
            with torch.inference_mode():
                trainer = pl.Trainer(
                    logger=None,
                    enable_progress_bar=True
                )
            test_preds = trainer.predict(mpnn, test_loader)
            test_preds = np.concatenate(test_preds, axis=0)
            pred_list.append(test_preds)
        return pred_list

    # Function that takes in a list of mlticomponent MPNN
    # model paths, a test dataframe, and a vector of molar fraction x
    # It the dataframe with mean and std of gamma predictions for the
    # solutes in 'solute_smiles_canonical' and solvents in 'solvent_smiles_canonical'
    # of the columns of the dataframe
    def predict(self,checkpoint_path_list,df_test,x):
        if self.mixture:
            pred_list = []
            df_mix = self._prepare_mixture_dataframe(df_test, x)
            for checkpoint_path in checkpoint_path_list:
                if '.pt' in str(checkpoint_path):
                    mcmpnn = MixtureMPNN.load_from_file(checkpoint_path,map_location=device)
                else:
                    mcmpnn = MixtureMPNN.load_from_checkpoint(checkpoint_path,map_location=device)
                smiles_columns = ["solute_smiles_canonical", "solvent1_smiles_canonical", "solvent2_smiles_canonical"]
                frac_columns = ["molefrac"]
                target_columns = ["MP_std"]
                smiss = df_mix.loc[:, smiles_columns].values
                fracs = df_mix.loc[:, frac_columns].copy()
                fracs["molefrac"] = fracs["molefrac"].fillna(1.0)
                fracs = fracs.values
                ys = df_mix.loc[:, target_columns].values
                extra_datapoint_descriptors = df_mix[["x", "Temperature [K]"]].values
                all_data = [[MoleculeDatapoint.from_smi(smis[0], y, x_d=X_d) for smis, y, X_d in zip(smiss, ys, extra_datapoint_descriptors)]]
                all_data += [[ComponentDatapoint.from_smi(smis[1], w_fp=f[0]) for smis, f in zip(smiss, fracs)]]
                all_data += [[ComponentDatapoint.from_smi(smis[2], w_fp=1-f[0]) if not self._missing_smiles(smis[2]) else ComponentDatapoint(None) for smis, f in zip(smiss, fracs)]]
                test_datasets = [
                    MoleculeDataset(all_data[0]),
                    ComponentDataset(all_data[1]),
                    ComponentDataset(all_data[2]),
                ]
                test_mcdset = MixtureDataset(test_datasets)
                test_loader = DataLoader(test_mcdset, batch_size=64, shuffle=False, collate_fn=collate_mixture)
                with torch.inference_mode():
                    trainer = pl.Trainer(
                        logger=None,
                        enable_progress_bar=True,
                        accelerator="auto",
                        devices=1,
                        deterministic=True
                    )
                    results = trainer.predict(mcmpnn, test_loader)
                test_preds = np.concatenate([t.numpy().flatten() for t in results])
                pred_list.append(test_preds)
            df_test['ln_gamma'] = np.array(pred_list).mean(axis=0)
            df_test['ln_gamma_std'] = np.array(pred_list).std(axis=0)
            return df_test
        else:
            pred_list = []
            for checkpoint_path in checkpoint_path_list:
                if '.pt' in str(checkpoint_path):
                    mcmpnn = multi.MulticomponentMPNN.load_from_file(checkpoint_path,map_location=device)
                else:
                    mcmpnn = multi.MulticomponentMPNN.load_from_checkpoint(checkpoint_path,map_location=device)
                smiles_columns = ['solute_smiles_canonical', 'solvent_smiles_canonical'] # name of the column containing SMILES strings
                smiss = df_test[smiles_columns].values
                n_componenets = len(smiles_columns)
                X_d = np.concatenate([x,df_test[['Temperature [K]']].to_numpy()],axis=1)
                test_datapointss = [[data.MoleculeDatapoint.from_smi(smi, x_d=X_d) for smi,X_d in zip(smiss[:, i],X_d)] for i in range(n_componenets)]
                featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
                test_dsets = [data.MoleculeDataset(test_datapoints, featurizer) for test_datapoints in test_datapointss]
                test_mcdset = data.MulticomponentDataset(test_dsets)
                test_loader = data.build_dataloader(test_mcdset, shuffle=False,batch_size=1)
                with torch.inference_mode():
                    trainer = pl.Trainer(
                        logger=None,
                        enable_progress_bar=True,
                        accelerator="auto",
                        devices=1,
                        deterministic=True
                    )
                    test_preds = trainer.predict(mcmpnn, test_loader)
                test_preds = np.concatenate(test_preds, axis=0)
                pred_list.append(test_preds)
            df_test['ln_gamma'] = np.array(pred_list)[:,:,0].mean(axis=0)
            df_test['ln_gamma_std'] = np.array(pred_list)[:,:,0].std(axis=0)
            return df_test  

    # Function to calculate molar fraction
    # It takes in the molar fraction x0 and
    # a dataframe that has columns for MP_pred,
    # dHfus_pred, Temperature [K], and gamma
    # It uses a default threshold of 0.99 as
    # maximum allowable mole fraction
    # it returns molar fraction x
    def calc_x_solid(self,x0,df,checkpoint_path_list):
        df = self.predict(checkpoint_path_list,df,pd.DataFrame(x0).to_numpy()) # predict gamma at x0
        x = np.exp(df['dHfus_pred']/(self.R) * ((1/df['MP_pred']) - (1/df['Temperature [K]'])) - (df['ln_gamma']) ) 
        x[x>self.thresh] = self.thresh
        return x,np.exp(df['ln_gamma'])

    # Function that calculates solid solubililty
    # it takes in a dataframe with columns
    # 'solute_smiles_canonical', 'solvent_smiles_canonical',
    # 'Temperature [K]', and 'solvent_density'
    # the function predicts MP based on trained
    # models. Then it iterates with an initial guess of gamma=0
    # to approximate solubility at saturation
    def calculate_solubility_solid(self,df):
        print('solid',df.shape[0])
        # Predict dHfus
        checkpoint_path_list = []
        for i in range(self.Num_ensembles):
            if self.mixture:
                checkpoint_path_list.append(self._mixture_model_path('dHfus_'+str(i)+'.pt'))
            else:
                checkpoint_path_list.append(self._model_path('dHfus', 'model_'+str(i)+'.pt'))
        preds = self.predict_single(checkpoint_path_list,df)
        df['dHfus_pred'] = np.array(preds)[:,:,0].T.mean(axis=1)
        df['dHfus_std'] = np.array(preds)[:,:,0].T.std(axis=1)
        # Calculate solubility
        checkpoint_path_list = []
        for i in range(self.Num_ensembles):
            if self.mixture:
                checkpoint_path_list.append(self._mixture_model_path('model_gamma_mix_FT_'+str(i*10)+'.pt'))
            else:
            # checkpoint_path_list.append('trained_models/gamma/model_'+str(i)+'.pt')
                checkpoint_path_list.append(self._model_path('gamma', 'model_final_'+str(i)+'.pt'))
        x = df[['dHfus_std']] * 0 # initialize at infinite dilution
        for i in range(self.N_iteration): # Iterate to adjust for x
            x,gamma = self.calc_x_solid(x,df,checkpoint_path_list)
            S = x * df['solvent_density']
            logS = np.log10(S)
        return logS,gamma

    # Function to calculate molar fraction
    # It takes in the molar fraction x1, x2,
    # and a dataframe. 
    # It uses a default threshold of 0.99 as
    # maximum allowable mole fraction
    # it returns molar fractions x1 and x2
    def calc_x_liquid(self,x1,x2,df,checkpoint_path_list):
        df = self.predict(checkpoint_path_list,df,pd.DataFrame(x2).to_numpy())
        gamma2 = np.exp(df['ln_gamma'])
        df = self.predict(checkpoint_path_list,df,pd.DataFrame(x1).to_numpy())
        gamma1 = np.exp(df['ln_gamma'])
        x1 = gamma2/gamma1 * x2.values.flatten()
        x1[x1>self.thresh] = self.thresh
        df = self.predict(checkpoint_path_list,df,pd.DataFrame(x1).to_numpy())
        gamma1 = np.exp(df['ln_gamma'])
        x2 = gamma1/gamma2 * x1.values.flatten()
        x2[x2>self.thresh] = self.thresh
        return x1,x2,gamma1,gamma2

    # Function that calculates liquid solubililty
    # it takes in a dataframe with columns
    # 'solute_smiles_canonical', 'solvent_smiles_canonical',
    # 'Temperature [K]', and 'solvent_density'
    # The function iterates with an initial guess of x1=0
    # and x2=1 to approximate solubility at saturation
    def calculate_solubility_liquid(self,df):
        print('liquid',df.shape[0])
        # Calculate solubility
        checkpoint_path_list = []
        for i in range(self.Num_ensembles):
            if self.mixture:
                checkpoint_path_list.append(self._mixture_model_path('model_gamma_mix_FT_'+str(i*10)+'.pt'))
            else:
            # checkpoint_path_list.append('trained_models/gamma/model_'+str(i)+'.pt')
                checkpoint_path_list.append(self._model_path('gamma', 'model_final_'+str(i)+'.pt'))
        x1 = df[['MP_std']] * 0.0 # initialize solvent-rich phase
        x2 = x1 + 1.0 # initialize solute-rich phase
        for i in range(self.N_iteration): # Iterate to adjust for x
            x1,x2,gamma1,gamma2 = self.calc_x_liquid(x1,x2,df,checkpoint_path_list)
            S = x1 * df['solvent_density']
            logS = np.log10(S)
        return logS,gamma1

    def calculate_solubility(self,df):
        # Use QSPR to fill in missing solvent densities (approximated at 298K)
        df = self._prepare_density_dataframe(df)
        if not self.mixture:
            df = fill_solvent_density(df)
        # Predict MP
        checkpoint_path_list = []
        for i in range(self.Num_ensembles):
            if self.mixture:
                checkpoint_path_list.append(self._mixture_model_path('MP_'+str(i)+'.pt'))
            else:
                checkpoint_path_list.append(self._model_path('MP', 'model_'+str(i)+'.pt'))
        preds = self.predict_single(checkpoint_path_list,df)
        df['MP_pred'] = np.array(preds)[:,:,0].T.mean(axis=1)
        df['MP_std'] = np.array(preds)[:,:,0].T.std(axis=1)
        # Split data to solid/liquid
        idx = df['MP_pred']>df['Temperature [K]']
        df_solid  = df[ idx].reset_index(drop=True)
        df_liquid = df[~idx].reset_index(drop=True)
        # Route to functions
        df['logS_calc'] = 0.0 # initalize
        df['gamma'] = 0.0 # initalize
        if df_solid.shape[0]>0:
            logS, gamma = self.calculate_solubility_solid(df_solid)
            df.loc[idx, ['logS_calc', 'gamma']] = np.vstack([logS, gamma]).T
            # df.loc[ idx,['logS_calc','gamma']] = self.calculate_solubility_solid(df_solid)
        if df_liquid.shape[0]>0:
            logS, gamma = self.calculate_solubility_liquid(df_liquid)
            df.loc[~idx,['logS_calc','gamma']] = np.vstack([logS, gamma]).T
            # df.loc[-idx,['logS_calc','gamma']] = self.calculate_solubility_liquid(df_liquid)
        df.to_csv('Results.csv',index=False)
        return df['logS_calc']
