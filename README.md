# MCMC for Bayesian Estimation of Differential Privacy from Membership Inference Attacks

This repository includes the implementation of the paper "MCMC for Bayesian estimation of Differential Privacy from Membership Inference Attacks" (https://www.arxiv.org/pdf/2504.16683). The code supports both artificial and real data experiments, and includes scripts for running attacks, estimating DP parameters, and visualizing results.


## File and Script Descriptions

This repository includes both the MATLAB and the Python version of the source code and experimental scripts. The figures and results presented in the paper were generated using these MATLAB codes. Additionally, we provide the Python version of the same codes to enhance reproducibility.

### Source Code (`src/`)
- **bivariate_log_pdf.m/.py**: Computes the log of the bivariate normal PDF for use in MCMC estimation.
- **DPinR.m/.py**: Determines if a point in [0, 1] x [0, 1] is in the region R(epsilon, delta) of admissible (alpha, beta) error probabilities.
- **gen_beta_DP.m/.py**: Generates (alpha, beta) pairs for a given (epsilon, delta)-DP constraint.
- **LRT_normal.m/.py**: Implements the likelihood ratio test (LRT) for comparing two normal distributions with a single observation.
- **MCMC_epsDP.m/.py**: Implements the basic MCMC-DP-Est algorithm (Algorithm 1).
- **MCMC_epsDP_v2.m/.py**: Implements the basic MCMC-DP-Est algorithm (Algorithm 1).
- **read_losses.m/.py**: Reads the files and stores the loss-based metrics in matrices.

### Experiment Scripts (`experiments/`)
- **main_artificial_exp1.m/.py**: Runs artificial experiments to analyze credible intervals (CI) for epsilon vs. attack strength (s) and sample size (N). Generates Figure 2 in the paper.
- **main_artificial_exp2.m/.py**: Runs artificial experiments for strong/weak attacks, generating Figure 3.
- **main_XYplots_and_DPEstMCMC.m/.py**: Main script for real data experiments. Generates Figures 4, 5, 6, and Table 1. Reads loss metrics, runs MCMC estimation, and produces plots/histograms.

### Analysis (`analysis/`)
- **main_measure_mia.py**: Applies the membership inference attack (MIA) for real data experiments (Section 4.2 of the paper). Trains models with/without DP noise, evaluates losses, and saves results for further analysis.

### Data (`data/`)
- **dp_attack_*.txt**: Output files containing loss metrics from MIA experiments under various DP/no-DP and weight initialization settings.
- **sample_info_n_1000.txt**: Stores indices and sample information for real data experiments (used by both MATLAB and Python scripts).

## Usage

### MATLAB
1. Add the `matlab/src/` directory to your MATLAB path:
    ```matlab
    addpath('src')
    ```
2. Run experiment scripts from the `matlab/experiments/` directory to reproduce results and figures from the paper.

### Python
1. Install required packages (TensorFlow, NumPy, SciPy).
2. Run `main_measure_mia.py` from the `analysis/` directory to generate MIA results for real data.
3. Run experiment scripts from the `python/experiments/` directory to reproduce results and figures from the paper.

### Data
- All experiment outputs and sample indices are stored in the `data/` directory. Experiment scripts read these files for analysis and plotting.

## Contact

For questions, please contact [cerenyildirim@sabanciuniv.edu](mailto:cerenyildirim@sabanciuniv.edu).

---

*Last updated: November 2025*