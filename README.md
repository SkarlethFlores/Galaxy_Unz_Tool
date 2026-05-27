# Galaxy Fine Structure Fitting Tool

A Python tool for fitting far-infrared fine structure emission lines to measure **gas-phase metallicity** in galaxies — free from the effects of dust extinction.

---

## Overview

This tool uses Bayesian inference to simultaneously fit:
- **Ionization parameter** (log U)
- **Gas-phase metallicity** (Z)
- **Electron density** (nₑ)

by modeling far-infrared fine structure emission line ratios. Because far-IR lines are unaffected by dust extinction, this approach is especially powerful for heavily obscured or high-redshift galaxies.

---

## Repository Structure

```
Galaxy_Unz_Tool/
├── data/                          # Reference model grids and calibration data
├── input/                         # Input spectra / line flux files
├── output/                        # Fitting results
├── Plots/                         # Output figures
├── gism_Functions.py              # Core fitting and utility functions
├── main-Caller.ipynb              # Main notebook to run the fitting pipeline
├── main-He2-10-explore_results.ipynb  # Example: results exploration for He 2-10
├── model-part-1.ipynb             # Model grid construction (Part 1)
└── requirements.txt               # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Dependencies listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/SkarlethFlores/Galaxy_Unz_Tool.git

# Navigate to the project directory
cd Galaxy_Unz_Tool

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The correct command is `pip install -r requirements.txt` (with the `-r` flag).

### Usage

1. Place your input line flux data in the `input/` folder.
2. Open `main-Caller.ipynb` in Jupyter.
3. Adjust the input parameters (source name, line fluxes, uncertainties) at the top of the notebook.
4. Run all cells — results will be saved to `output/` and plots to `Plots/`.

For an example of how to explore and interpret results, see `main-He2-10-explore_results.ipynb`.

---

## Features

- Bayesian parameter inference (ionization potential, metallicity, electron density) from far-IR fine structure lines
- Extinction-independent metallicity measurement
- Modular design — core functions are in `gism_Functions.py` for easy reuse
- Example notebook with a real galaxy (He 2-10)

---

## Citation / Attribution

If you use this code in your research, please acknowledge it by including the following in your paper or documentation:

```
This work made use of the Galaxy Fine Structure Fitting Tool developed by Skarleth Motiño Flores,
available at https://github.com/SkarlethFlores/Galaxy_Unz_Tool.
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE). You are free to use, modify, and distribute this code with proper attribution.

---

*Developed by [Skarleth Motiño Flores](https://github.com/SkarlethFlores)*
