# Project 3: Neutron Transport Monte Carlo

This repository contains an anonymised computational physics project on Monte
Carlo neutron transport through shielding materials.
The main notebook is the latest completed project notebook, including saved
outputs, explanatory text, the Woodcock tracking stretch task, and the final
runtime check.

An optional scientific-machine-learning extension is also included as a separate
notebook. It uses simulation data to train surrogate models for fast
transmission prediction.

## Contents

- `notebooks/neutron_transport_monte_carlo_with_outputs.ipynb` - latest main
  project notebook with saved outputs.
- `notebooks/neutron_transport_monte_carlo_clean.ipynb` - output-free copy of
  the latest main notebook for lighter version control.
- `notebooks/surrogate_model_extension.ipynb` - optional neural-network
  surrogate modelling extension.
- `notebooks/surrogate_model_extension_clean.ipynb` - output-free copy of the
  extension notebook.
- `src/neutron_transport.py` - reusable Monte Carlo simulation and plotting
  utilities extracted for reproducible local runs.
- `scripts/run_neutron_transport.py` - companion script that regenerates CSV
  summaries and SVG figures in `results/`.
- `results/` - generated CSV summaries, diagnostics, and SVG plots.

## Project Summary

The main notebook builds the neutron-transport simulation step by step. It first
checks the random-number tools: uniform sampling, three-dimensional point clouds,
exponential free paths, isotropic unit vectors, and isotropic random steps. It
then converts microscopic cross-sections for water, lead, and graphite into
macroscopic material properties.

The transport model simulates thermal neutrons entering a slab at `x = 0`.
Neutrons undergo free flight, absorption, isotropic scattering, reflection from
the entrance side, or transmission through the far side. The notebook estimates
absorption, reflection, and transmission probabilities for 10 cm slabs, scans
slab thickness, fits effective transmission attenuation lengths, and validates a
Woodcock tracking method for two adjacent slabs.

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

## Reproduce Scripted Results

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
random seed, neutron counts, or package versions are changed. The script is a
reproducibility companion to the notebook, so its generated values may not match
the saved notebook outputs digit-for-digit.

## Notebook Headline Results

The latest saved notebook outputs include:

- Water absorption-only attenuation length: `45.01 +/- 0.12 cm`, compared with
  the theoretical value `44.97 cm`.
- For a `10 cm` slab:
  - Water: `A = 0.1997`, `R = 0.7972`, `T = 0.0031`
  - Lead: `A = 0.1015`, `R = 0.6186`, `T = 0.2800`
  - Graphite: `A = 0.0080`, `R = 0.6863`, `T = 0.3057`
- Fitted effective transmission attenuation lengths:
  - Water: `1.89 +/- 0.03 cm`
  - Lead: `9.06 +/- 0.24 cm`
  - Graphite: `11.27 +/- 0.45 cm`
- Woodcock validation against direct 20 cm slab simulations gives z-scores below
  1 for water, lead, and graphite.
- Mixed two-slab Woodcock transmission is about `0.143` for lead then graphite
  and about `0.141` for graphite then lead.

## Notes for Upload

The clean notebooks are recommended for normal commits because they keep Git
diffs readable. The saved-output notebook is retained so repository viewers can
render a complete view of the analysis.
