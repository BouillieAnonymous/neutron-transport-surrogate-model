# Neutron Transport Monte Carlo and Surrogate Model

This repository contains a Monte Carlo notebook on neutron transport through
shielding materials, plus an extension notebook that trains surrogate models for
fast transmission prediction.

## Contents

- `notebooks/neutron_transport_monte_carlo_with_outputs.ipynb` - main
  project notebook with saved outputs.
- `notebooks/neutron_transport_monte_carlo_clean.ipynb` - the same main notebook with
  outputs removed for lighter version control.
- `notebooks/surrogate_model_extension.ipynb` - extension notebook for the
  neural-network surrogate model.
- `notebooks/surrogate_model_extension_clean.ipynb` - output-free version of
  the extension notebook.
- `src/neutron_transport.py` - reusable Monte Carlo simulation and plotting
  utilities.
- `scripts/run_neutron_transport.py` - script that regenerates the numerical tables and
  SVG figures.
- `results/` - generated CSV summaries, diagnostics, and SVG plots.

## Project Summary

The main notebook implements random number diagnostics, isotropic scattering,
exponential free paths, random walks through slabs of water, lead, and graphite,
and a Woodcock tracking stretch task for adjacent slabs. It reports absorption,
reflection, and transmission rates as functions of slab thickness.

The extension notebook builds synthetic simulation data and compares two
surrogate regressors:

- `sklearn.neural_network.MLPRegressor`
- a small PyTorch feedforward network

Both models are trained to approximate Monte Carlo transmission estimates much
faster than repeatedly running the full random-walk simulation.

## Setup

Create a Python environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Reproduce Results

From the repository root:

```powershell
python scripts/run_neutron_transport.py
```

This regenerates the files in `results/`, including:

- `material_properties.csv`
- `slab_10cm_rates.csv`
- `thickness_scan_rates.csv`
- `attenuation_lengths.csv`
- `woodcock_two_slab_rates.csv`
- representative SVG plots

Because the simulation is Monte Carlo, final digits may change slightly if the
random seed, neutron counts, or package versions are changed.

## Headline Results

Current generated results include:

- Water no-scattering attenuation length: about `44.6 +/- 0.1 cm`
- For a `10 cm` slab:
  - Water: `A = 0.206`, `R = 0.791`, `T = 0.004`
  - Lead: `A = 0.101`, `R = 0.621`, `T = 0.278`
  - Graphite: `A = 0.005`, `R = 0.687`, `T = 0.308`
- Fitted transmission attenuation lengths:
  - Water: `1.92 +/- 0.05 cm`
  - Lead: `9.78 +/- 0.09 cm`
  - Graphite: `12.74 +/- 0.13 cm`

## Notes for GitHub Upload

The output-free notebooks are recommended for normal commits because they keep
Git diffs readable. The saved-output notebook is retained so GitHub can render a
complete view of the analysis.
