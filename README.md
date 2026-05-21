# Fusion-Cycle

### Overview

This repository includes machine learning tools to estimate solubility.

For solid solubility, the following equation will be used

$$S_T\approx\rho_{solvent}x^T_{sat}\approx\frac{\rho_{solvent}}{\gamma^T_{sat}}\exp\left[\frac{\Delta H_{fus}}{R}\left(\frac{1}{T_{mp}}-\frac{1}{T}\right)\right]$$

to iteratively calculate solid solubility based on predictions of $\Delta H_{fus}$, $T_{mp}$, and $\gamma_x^T$ coming from three individual ML models, and $\rho_{solvent}$ which is assumed to be known.

For liquids we solve for the solubility of the liquid solute in the solvent-rich phase by iteratively solving the following equation:

$$x^\alpha_B \gamma^\alpha_B = x^\beta_B \gamma^\beta_B$$

where $x$ is the molar fraction, $\gamma$ is the activity coefficient, $B$ is the solute, $\alpha$ is the solvent-rich phase, and $\beta$ is the solute-rich phase (the derivation of this equation is shown in the supporting information). We use an initial guess of $x^\alpha_B=0$ and $x^\beta_B=1$ to calculate $x^\alpha_B$, and then update the prediction of $\gamma^\alpha_B$ to calculate $x^\beta_B$.

<div align="center">
  <img width="729" height="493" alt="process" src="https://github.com/user-attachments/assets/6b39af2e-a88e-492e-8451-516dee2859d2" />
</div>

### 1. Requirements

* chemprop (the official [Chemprop](https://github.com/chemprop/chemprop) package)

Install via:

```bash
pip install chemprop
```

---

### 2. Installation

#### Local editable install

For development, clone this repository and install it into an existing environment:

```bash
git clone https://github.com/emadalibrahim/Fusion-Cycle.git
cd Fusion-Cycle
python -m pip install -e .
```

#### Build and install a conda package

The repository includes a conda-build recipe in `conda-recipe/`. Build the package locally with:

```bash
conda install -c conda-forge conda-build
conda build conda-recipe -c conda-forge -c pytorch
```

Then install the locally built package:

```bash
conda install --use-local fusioncycle -c conda-forge -c pytorch
```

The conda package includes the trained model files under `trained_models/`, so it is large. After installation, both import styles are supported:

```python
import Fusion_Cycle

model = Fusion_Cycle.model()
```

or:

```python
import fusioncycle

model = fusioncycle.model()
```

---

### 3. Data Formatting

#### Single-solvent predictions

For the default single-solvent model (`mixture=False`), solubility calculation expects the following columns:

* **Solute/Solvent**
  columns names: `solute_smiles_canonical` and `solvent_smiles_canonical`.
* **Temperature**
  column name: `Temperature [K]`. A numeric column (e.g. `298.15`) in [Kelvin].
* **Density**
  column name: `solvent_density`. A numeric column (e.g. `17.25`) in [mol/L].

If `solvent_density` is missing or empty, a two-parameter QSPR model is used to estimate the solvent density, approximated at 298 K.

#### Mixture predictions

For binary solvent mixtures, initialize the model with `mixture=True`. The mixture model expects:

* **Solute**
  column name: `solute_smiles_canonical`.
* **Solvents**
  column names: `solvent1_smiles_canonical` and `solvent2_smiles_canonical`.
  If the row is a single-solvent case, `solvent2_smiles_canonical` may be empty.
* **Mixture composition**
  column name: `molefrac`, the mole fraction of solvent 1. Solvent 2 is assigned `1 - molefrac`.
* **Temperature**
  column name: `Temperature [K]`, in [Kelvin].
* **Density**
  preferred column name: `solvent_avg_density`, in [mol/L].

For `mixture=True`, `solvent_avg_density` is used directly when present. If it is missing, Fusion-Cycle estimates missing component densities with the QSPR model and computes the average mixture density as:

$$\rho_{avg} = \frac{1}{\left(\frac{x_1}{\rho_1}\right) + \left(\frac{x_2}{\rho_2}\right)}$$

where `x1 = molefrac`, `x2 = 1 - molefrac`, `rho1 = solvent1_density`, and `rho2 = solvent2_density`. If `solvent1_density` and/or `solvent2_density` are already provided, those values are used and only missing component densities are estimated.

Mixture model weights are loaded from:

* `trained_models/Mixture_models/Full` when `segment=False`
* `trained_models/Mixture_models/Segment` when `segment=True`

---

### 4. Example

First, clone this repository and enter the directory:

```bash
git clone https://github.com/emadalibrahim/Fusion-Cycle.git
cd Fusion-Cycle
```
An example run is shown in `Example_run.ipynb`

Single-solvent usage:

```python
import pandas as pd
import Fusion_Cycle

df = pd.DataFrame(
    {
        "solute_smiles_canonical": ["CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1"],
        "solvent_smiles_canonical": ["CCO"],
        "Temperature [K]": [298.15],
        "solvent_density": [17.06],
    }
)

model = Fusion_Cycle.model(N_iteration=10, thresh=0.99, Num_ensembles=5, mixture=False)
logS = model.calculate_solubility(df)
```

Binary-mixture usage:

```python
import pandas as pd
import Fusion_Cycle

df = pd.DataFrame(
    {
        "solute_smiles_canonical": ["CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1"],
        "solvent1_smiles_canonical": ["CCO"],
        "solvent2_smiles_canonical": ["O"],
        "molefrac": [0.5],
        "Temperature [K]": [298.15],
        "solvent_avg_density": [30.0],
    }
)

model = Fusion_Cycle.model(
    N_iteration=10,
    thresh=0.99,
    Num_ensembles=5,
    mixture=True,
    segment=False,
)
logS = model.calculate_solubility(df)
```
