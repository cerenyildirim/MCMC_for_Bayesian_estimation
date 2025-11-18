# MCMC for Bayesian Estimation of Differential Privacy from Membership Inference Attacks

This repository includes the implementation of the paper "MCMC for Bayesian estimation of Differential Privacy from Membership Inference Attacks" ```markdown
[Paper](https://link.springer.com/chapter/10.1007/978-3-032-06096-9_23)```. The code supports both artificial and real data experiments, and includes scripts for running attacks, estimating DP parameters, and visualizing results.


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


## Citation 

If you use this code in your research, please cite:
```bibtex
@InProceedings{10.1007/978-3-032-06096-9_23,
    author="Y{\i}ld{\i}r{\i}m, Ceren
    and Kaya, Kamer
    and Y{\i}ld{\i}r{\i}m, Sinan
    and Sava{\c{s}}, Erkay",
    editor="Ribeiro, Rita P.
    and Pfahringer, Bernhard
    and Japkowicz, Nathalie
    and Larra{\~{n}}aga, Pedro
    and Jorge, Al{\'i}pio M.
    and Soares, Carlos
    and Abreu, Pedro H.
    and Gama, Jo{\~a}o",
    title="MCMC for Bayesian Estimation of Differential Privacy from Membership Inference Attacks",
    booktitle="Machine Learning and Knowledge Discovery in Databases. Research Track",
    year="2026",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="397--414",
    abstract="We propose a new framework for Bayesian estimation of differential privacy, incorporating evidence from multiple membership inference attacks (MIA). Bayesian estimation is carried out via a Markov Chain Monte Carlo (MCMC) algorithm, named MCMC-DP-Est, which provides an estimate of the full posterior distribution of the privacy parameter (e.g., instead of just credible intervals). Critically, the proposed method does not assume that privacy auditing is performed with the most powerful attack on the worst-case (dataset, challenge point) pair, which is typically unrealistic. Instead, MCMC-DP-Est jointly estimates the strengths of MIAs used and the privacy of the training algorithm, yielding a more cautious privacy analysis. We also present an economical way to generate measurements for the performance of an MIA that is to be used by the MCMC method to estimate privacy. We present the use of the methods with numerical examples with both artificial and real data.",
    isbn="978-3-032-06096-9"
}
```

## Contact

For questions, please contact [cerenyildirim@sabanciuniv.edu](mailto:cerenyildirim@sabanciuniv.edu).

---

*Last updated: November 2025*