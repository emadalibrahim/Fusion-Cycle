# Implements a QSPR model for liquid densities based on the Karelson & Perkson (1999) paper.
# This script calculates two key descriptors:
# 1. Intrinsic density (ρR)
# 2. Electrostatic interaction per atom (Eelstat), using an approximation

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import pickle
import os

# --- van der Waals radii from paper (Å) ---
# Note: The original paper used different radii for different hybridization states.
# We use a single, representative value for simplicity.
vdw_radii = {
    "H": 1.20,
    "C": 1.70,  # Using sp3 value as a general representative
    "N": 1.55,  # Using sp3 value as a general representative
    "O": 1.52,  # Using sp2 value as a general representative
    "S": 1.80,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.97,
}

def intrinsic_density(mol):
    """
    Computes intrinsic density (ρR) = Molecular Weight / Molecular Volume.
    The volume is approximated as the sum of non-overlapping van der Waals spheres.
    The original paper used a more complex method for calculating overlapping spheres.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        float: The calculated intrinsic density.
    """
    if not mol:
        return np.nan
    
    mw = Descriptors.ExactMolWt(mol)
    volume = 0.0
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        r = vdw_radii.get(elem, 1.7)  # Fallback to C radius if element not found
        volume += (4.0 / 3.0) * np.pi * r**3
    
    if volume == 0:
        return np.nan
        
    return mw / volume

def electrostatic_energy_per_atom(mol):
    """
    Approximates the total molecular electrostatic interaction per atom (Eelstat).
    The original paper used a semi-empirical quantum chemistry method (AM1).
    This implementation uses RDKit's Gasteiger charges as a practical proxy,
    summing the squares of the charges and dividing by the number of atoms.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        float: The approximated electrostatic descriptor.
    """
    if not mol:
        return np.nan

    ComputeGasteigerCharges(mol)
    charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
    
    if not charges:
        return np.nan
        
    sum_sq_charges = np.sum(np.square(charges))
    num_atoms = mol.GetNumAtoms()
    
    return sum_sq_charges / num_atoms

def build_features(df):
    """
    Builds the feature matrix and adds the descriptor columns to the DataFrame.
    This function processes each compound using its SMILES string to calculate
    the two descriptors for the QSPR model.

    Args:
        df (pandas.DataFrame): The input DataFrame which must contain a 'SMILES' column.

    Returns:
        pandas.DataFrame: The DataFrame with 'rho_R' and 'Eelstat' columns added.
    """
    df['rho_R'] = np.nan
    df['Eelstat'] = np.nan
    
    for index, row in df.iterrows():
        smiles = row.get('SMILES')
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                print(f"Could not generate molecule for SMILES: {smiles}")
                continue

            df.at[index, 'rho_R'] = intrinsic_density(mol)
            df.at[index, 'Eelstat'] = electrostatic_energy_per_atom(mol)
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            
    return df

def train_and_evaluate_qspr(X, y):
    """
    Trains a linear regression model and evaluates its performance.

    Args:
        X (np.array): Feature matrix.
        y (np.array): Target vector.

    Returns:
        tuple: A tuple containing the trained model and evaluation results.
    """
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return model, r2, rmse

def save_model(model, filename):
    """Saves a trained model to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to '{filename}'.")

def load_model(filename):
    """Loads a saved model from a file using pickle."""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from '{filename}'.")
    return model


def fill_solvent_density(df, model_path='qspr_density_model.pkl'):
    """
    Fills gaps in the 'solvent_density' column of a DataFrame using a QSPR model.

    Args:
        df (pd.DataFrame): The input DataFrame, which must contain columns
                          necessary for the QSPR model's features (e.g., 'solvent_smiles').
        model_path (str): The file path to the saved QSPR model.

    Returns:
        pd.DataFrame: The DataFrame with the 'solvent_density' column filled.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_path)

    
    df['SMILES'] = df['solvent_smiles_canonical']

    if 'solvent_density' not in df.columns:
        df['solvent_density'] = np.nan

    # Find the indices of rows with missing solvent density
    missing_idx = df['solvent_density'].isna()

    if missing_idx.sum()==0:
        return df
    
    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'. Cannot fill gaps.")
        return df
        
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from '{model_path}'.")
    except Exception as e:
        print(f"Error loading model: {e}. Cannot fill gaps.")
        return df

    # Only process the rows that need filling
    df_to_predict = df.loc[missing_idx].copy()
            
    df_to_predict = build_features(df_to_predict)
    
    # Get the features for prediction
    X_predict = df_to_predict[['rho_R', 'Eelstat']].values

    if len(X_predict) > 0:
        # Predict the missing values
        predicted_densities = model.predict(X_predict)
        
        # Fill the original DataFrame using the indices and convert to mol/L
        df_to_predict['solvent_MW'] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in df_to_predict['SMILES']]
        df.loc[missing_idx, 'solvent_density'] = (predicted_densities / df_to_predict['solvent_MW']) * 1000
        print(f"Successfully filled {len(predicted_densities)} missing values.")
    else:
        print("No valid data to predict for the missing rows.")
        
    return df

if __name__ == "__main__":
    # Define a filename for the saved model
    MODEL_FILENAME = 'qspr_density_model.pkl'

    # --- Training and Saving the Model ---
    
    print("--- Phase 1: Training and Saving the Model ---")
    
    # Load the complete dataset
    df_train = pd.read_csv('density_data.csv')
    print("Dataset loaded with", len(df_train), "compounds.")

    # Build the feature matrix
    df_train = build_features(df_train)
    
    # Drop rows with NaN values in the descriptor columns before training
    df_clean = df_train.dropna(subset=['rho_R', 'Eelstat'])
    
    X_train = df_clean[['rho_R', 'Eelstat']].values
    y_train = df_clean['rexp'].values
    
    print("\nSuccessfully calculated descriptors for", len(X_train), "compounds.")

    # Train and evaluate the QSPR model
    model, r2, rmse = train_and_evaluate_qspr(X_train, y_train)

    # Print model details and evaluation results
    print("\n--- QSPR Model Training Results ---")
    print(f"Number of compounds used for training: {len(y_train)}")
    print(f"Coefficients (ρR, Eelstat): {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Save the trained model
    save_model(model, MODEL_FILENAME)
    
    print("\n" + "="*50 + "\n")

    
