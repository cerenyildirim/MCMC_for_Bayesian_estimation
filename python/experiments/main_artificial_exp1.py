# main_artificial_exp1.py

import numpy as np
import matplotlib.pyplot as plt
import time
from src.MCMC_epsDP import MCMC_epsDP

def run_experiment_1():
    """
    Main script for generating the first set of experiments in Section 4.1.
    (Corresponds to Figure 2 in the paper)
    """
    print("Starting experiment: Credible Intervals for Epsilon")
    # Set up matplotlib for LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12
    })

    # --- Parameters ---
    np.random.seed(1)
    M = 1_000_000  # MCMC iterations
    K = 100       # Number of auxiliary variables
    delta_DP = 0.01
    sigma_qe = 0.1  # Proposal std for epsilon
    log_norm_var = 10 # Prior variance of log epsilon

    # Experiment 1: CI vs. s
    s_vec = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99])
    L_s = len(s_vec)
    N_00, N_01 = 500, 500
    X, Y = 200, 200

    # Experiment 2: CI vs. N
    N_vec = np.arange(100, 1001, 100)
    L_N = len(N_vec)
    s_fixed = 0.99

    eps_DP_low = [np.zeros(L_s), np.zeros(L_N)]
    eps_DP_high = [np.zeros(L_s), np.zeros(L_N)]

    # --- Run Experiment 1: CI vs s (fixed N) ---
    print("\nRunning Experiment 1: CI vs. s...")
    start_time = time.time()
    for j, s in enumerate(s_vec):
        print(f"  s = {s} ({j+1}/{L_s})")
        eps_DP_samps, _ = MCMC_epsDP(
            np.array([N_00]), np.array([N_01]), np.array([X]), np.array([Y]),
            eps_DP=1.0, delta_DP=delta_DP, M=M, K=K, sigma_qe=sigma_qe,
            sigma_qs=0.01, log_norm_var=log_norm_var, s=s, sa=1, sb=1, est_s=0
        )
        eps_DP_after_burn_in = eps_DP_samps[M // 10:]
        eps_DP_low[0][j] = np.quantile(eps_DP_after_burn_in, 0.05)
        eps_DP_high[0][j] = np.quantile(eps_DP_after_burn_in, 0.95)
    print(f"Experiment 1 finished in {time.time() - start_time:.2f} seconds.")

    # --- Run Experiment 2: CI vs N (fixed s) ---
    print("\nRunning Experiment 2: CI vs. N...")
    start_time = time.time()
    for i, N in enumerate(N_vec):
        print(f"  N = {N} ({i+1}/{L_N})")
        N0, N1 = N, N
        X_n, Y_n = round(N0 * 0.4), round(N1 * 0.4)
        eps_DP_samps, _ = MCMC_epsDP(
            np.array([N0]), np.array([N1]), np.array([X_n]), np.array([Y_n]),
            eps_DP=1.0, delta_DP=delta_DP, M=M, K=K, sigma_qe=sigma_qe,
            sigma_qs=0.01, log_norm_var=log_norm_var, s=s_fixed, sa=1, sb=1, est_s=0
        )
        eps_DP_after_burn_in = eps_DP_samps[M // 10:]
        eps_DP_low[1][i] = np.quantile(eps_DP_after_burn_in, 0.05)
        eps_DP_high[1][i] = np.quantile(eps_DP_after_burn_in, 0.95)
    print(f"Experiment 2 finished in {time.time() - start_time:.2f} seconds.")


    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: CI vs s
    ax1.plot(s_vec, eps_DP_low[0], 'o-', label='5% Lower Bound')
    ax1.plot(s_vec, eps_DP_high[0], 's-', label='95% Upper Bound')
    ax1.set_title(f'Attack Sample Size: {N_00}+{N_01}, $X=Y={X}$')
    ax1.set_xlabel('$s$: Strength Coef. for Attack Performance Prior')
    ax1.set_ylabel('$90\%$ Credible Interval for $\epsilon$')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: CI vs N
    ax2.plot(N_vec, eps_DP_low[1], 'o-', label='5% Lower Bound')
    ax2.plot(N_vec, eps_DP_high[1], 's-', label='95% Upper Bound')
    ax2.set_title(f'$s = {s_fixed}$')
    ax2.set_xlabel('Sample Size for Attack ($N_0=N_1$)')
    ax2.set_ylabel('$90\%$ Credible Interval for $\epsilon$')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("figure2_artificial_exp1.pdf")
    plt.show()

if __name__ == '__main__':
    run_experiment_1()