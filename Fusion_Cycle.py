import pandas as pd
import numpy as np
import os
import torch
from lightning import pytorch as pl
from pathlib import Path
from chemprop import data, featurizers,models
from chemprop.models import multi

# Function that takes in a list of single molecule
# model paths and a test dataframe
# It returns a list of predictions for the smiles in
# the 'solute_smiles_canonical' columns of the dataframe
def predict_single(checkpoint_path_list,df_test):
    pred_list = []
    for checkpoint_path in checkpoint_path_list:
        if '.ckpt' in str(checkpoint_path):
            mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
        else:
            mpnn = models.MPNN.load_from_file(checkpoint_path)
        smiles_columns = 'solute_smiles_canonical' # name of the column containing SMILES strings
        smis = df_test[smiles_columns].values
        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
        test_loader = data.build_dataloader(test_dset, shuffle=False)
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True
                # accelerator="cpu",
                # devices=1
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
def predict(checkpoint_path_list,df_test,x):
    pred_list = []
    for checkpoint_path in checkpoint_path_list:
        if '.pt' in str(checkpoint_path):
            mcmpnn = multi.MulticomponentMPNN.load_from_file(checkpoint_path)
        else:
            mcmpnn = multi.MulticomponentMPNN.load_from_checkpoint(checkpoint_path)
        smiles_columns = ['solute_smiles_canonical', 'solvent_smiles_canonical'] # name of the column containing SMILES strings
        smiss = df_test[smiles_columns].values
        n_componenets = len(smiles_columns)
        X_d = np.concatenate([x,df_test[['Temperature [K]']].to_numpy()],axis=1)
        test_datapointss = [[data.MoleculeDatapoint.from_smi(smi, x_d=X_d) for smi,X_d in zip(smiss[:, i],X_d)] for i in range(n_componenets)]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dsets = [data.MoleculeDataset(test_datapoints, featurizer) for test_datapoints in test_datapointss]
        test_mcdset = data.MulticomponentDataset(test_dsets)
        test_loader = data.build_dataloader(test_mcdset, shuffle=False)
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True,
                accelerator="auto",
                devices=1
            )
            test_preds = trainer.predict(mcmpnn, test_loader)
        test_preds = np.concatenate(test_preds, axis=0)
        pred_list.append(test_preds)
    df_test['gamma'] = np.array(pred_list)[:,:,0].mean(axis=0)
    df_test['gamma_std'] = np.array(pred_list)[:,:,0].std(axis=0)
    return df_test

# Function to calculate molar fraction
# It takes in the molar fraction x0 and
# a dataframe that has columns for MP_pred,
# dHfus_pred, Temperature [K], and gamma
# It uses a default threshold of 0.99 as
# maximum allowable mole fraction
# it returns molar fraction x
def calc_x(x0,df,checkpoint_path_list,thresh=0.99):
    R = 1.98720425864083/1000 # Kcal K-1 mol-1
    df = predict(checkpoint_path_list,df,pd.DataFrame(x0).to_numpy()) # predict gamma at x0
    x = np.exp(df['dHfus_pred']/(R) * ((1/df['MP_pred']) - (1/df['Temperature [K]'])) - (df['gamma']) ) 
    x[df['MP_pred']<df['Temperature [K]']] = 1/np.exp(df['gamma'][df['MP_pred']<df['Temperature [K]']]) # If T<MP neglect fusion term
    x[x>0.99] = 0.99
    return x

# Function that calculates solubililty
# it takes in a dataframe with columns
# 'solute_smiles_canonical', 'solvent_smiles_canonical',
# 'Temperature [K]', and 'solvent_density'
# the function predicts dHfus and MP based on trained
# models. Then it iterates with an initial guess of gamma=0
# to approximate solubility at saturation
def calculate_solubility(df,N_iteration):
    # Predict dHfus
    checkpoint_path_list = []
    for i in range(5):
        checkpoint_path_list.append('trained_models/dHfus/model_'+str(i)+'.pt')
    preds = predict_single(checkpoint_path_list,df)
    df['dHfus_pred'] = np.array(preds)[:,:,0].T.mean(axis=1)
    df['dHfus_std'] = np.array(preds)[:,:,0].T.std(axis=1)
    # Predict MP
    checkpoint_path_list = []
    for i in range(5):
        checkpoint_path_list.append('trained_models/MP/model_'+str(i)+'.pt')
    preds = predict_single(checkpoint_path_list,df)
    df['MP_pred'] = np.array(preds)[:,:,0].T.mean(axis=1)
    df['MP_std'] = np.array(preds)[:,:,0].T.std(axis=1)
    # Calculate solubility
    checkpoint_path_list = []
    for i in range(5):
        checkpoint_path_list.append('trained_models/gamma/model_'+str(i)+'.pt')
    x = df[['dHfus_std']] * 0 # initialize at infinite dilution
    for i in range(N_iteration): # Iterate to adjust for x
        x = calc_x(x,df,checkpoint_path_list)
        S = x * df['solvent_density']
        logS = np.log10(S)
        
    return logS