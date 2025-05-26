# MCMC for Bayesian Estimation of Differential Privacy from Membership Inference Attacks

This repository contains MATLAB and Python code for Bayesian estimation of differential privacy (DP) parameters using Markov Chain Monte Carlo (MCMC) methods, with a focus on analyzing membership inference attacks (MIA). The code supports both artificial and real data experiments, and includes scripts for running attacks, estimating DP parameters, and visualizing results.

## File and Script Descriptions

### MATLAB Source Code (`src/`)
- **bivariate_log_pdf.m**: Computes the log of the bivariate normal PDF for use in MCMC estimation.
- **DPinR.m**: Determines if a point in [0, 1] x [0, 1] is in the region R(epsilon, delta) of admissible (alpha, beta) error probabilities.
- **gen_beta_DP.m**: Generates (alpha, beta) pairs for a given (epsilon, delta)-DP constraint.
- **LRT_normal.m**: Implements the likelihood ratio test (LRT) for comparing two normal distributions with a single observation.
- **MCMC_epsDP.m**: Implements the basic MCMC-DP-Est algorithm (Algorithm 1).
- **MCMC_epsDP_v2.m**: Implements the basic MCMC-DP-Est algorithm (Algorithm 1).
- **read_losses.m**: Reads the files and stores the loss-based metrics in matrices.

### MATLAB Experiment Scripts (`experiments/`)
- **main_artificial_exp1.m**: Runs artificial experiments to analyze credible intervals (CI) for epsilon vs. attack strength (s) and sample size (N). Generates Figure 2 in the paper.
- **main_artificial_exp2.m**: Runs artificial experiments for strong/weak attacks, generating Figure 3.
- **main_XYplots_and_DPEstMCMC.m**: Main script for real data experiments. Generates Figures 4, 5, 6, and Table 1. Reads loss metrics, runs MCMC estimation, and produces plots/histograms.

### Python Analysis (`analysis/`)
- **main_measure_mia.py**: Applies the membership inference attack (MIA) for real data experiments (Section 4.2 of the paper). Trains models with/without DP noise, evaluates losses, and saves results for further analysis.

### Data (`data/`)
- **dp_attack_*.txt**: Output files containing loss metrics from MIA experiments under various DP/no-DP and weight initialization settings.
- **sample_info_n_1000.txt**: Stores indices and sample information for real data experiments (used by both MATLAB and Python scripts).

## Usage

### MATLAB
1. Add the `src/` directory to your MATLAB path:
    ```matlab
    addpath('src')
    ```
2. Run experiment scripts from the `experiments/` directory to reproduce results and figures from the paper.

### Python
1. Install required packages (TensorFlow, NumPy, SciPy).
2. Run `main_measure_mia.py` from the `analysis/` directory to generate MIA results for real data.

### Data
- All experiment outputs and sample indices are stored in the `data/` directory. MATLAB scripts read these files for analysis and plotting.

## References

This codebase implements methods and experiments from:
> "MCMC for Bayesian estimation of Differential Privacy from Membership Inference Attacks"

## Contact

For questions, please contact [cerenyildirim@sabanciuniv.edu](mailto:cerenyildirim@sabanciuniv.edu).

---

*Last updated: May 2025*