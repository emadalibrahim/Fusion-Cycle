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

* Python 3.8+
* RDKit
* pandas, numpy
* chemprop (the official [Chemprop](https://github.com/chemprop/chemprop) package)

Install via:

```bash
pip install rdkit-pypi pandas numpy chemprop
```

---

### 2. Data Formatting

Solubility calculation expects the following columns:

* **Solute/Solvent**
  columns names: `solute_smiles_canonical` and `solvent_smiles_canonical`.
* **Temperature**
  columns name: `Temperature [K]`. A numeric column (e.g. `298.15`) in [Kelvins].
* **Density**
  columns name: `solvent_density`. A numeric column (e.g. `17.25`) in [mol/L].

---

### 3. Example

An example run is shown in Example_run.ipynb
