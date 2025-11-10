# main_XYplots_and_DPEstMCMC.py
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from statsmodels.tsa.stattools import acf

# Import translated functions
from src.read_losses import read_losses
from src.LRT_normal import LRT_normal
from src.MCMC_epsDP_v2 import MCMC_epsDP_v2

def run_real_data_experiments():
    """
    This script generates Figures 4, 5, 6 and Table 1 from the paper.
    It requires the specified data files to be in the same directory.
    """
    print("Starting real data experiments...")
    # Set up matplotlib for LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10
    })

    # --- Setup and Initialization ---
    np.random.seed(1)
    foldername = 'Output_Figures_Python'
    os.makedirs(foldername, exist_ok=True)

    # Files where the loss metrics are stored
    filenames = [
        'dp_attack_wo_DP_fixed_weights_0.1.txt',
        'dp_attack_wo_DP_random_weights_0.1.txt',
        'dp_attack_w_DP_fixed_weights_0.1.txt',
        'dp_attack_w_DP_random_weights_0.1.txt',
        'dp_attack_w_DP_random_weights_0.05.txt',
        'dp_attack_w_DP_random_weights_0.01.txt'
    ]
    # Check if data files exist before proceeding
    for f in filenames:
        if not os.path.exists(f'../data/{f}'):
            print(f"Error: Data file '{f}' not found. This script cannot run without it.")
            print("Please place the required data files in the same directory.")
            return

    alg_names = [
        r'$\mathcal{A}_{1}$', r'$\mathcal{A}_{2}$',
        r'$\mathcal{A}_{3}$, $\sigma = 0.1$',
        r'$\mathcal{A}_{4}$, $\sigma = 0.1$',
        r'$\mathcal{A}_{4}$, $\sigma = 0.05$',
        r'$\mathcal{A}_{4}$, $\sigma = 0.01$'
    ]
    L_f = len(filenames)

    # MCMC parameters
    M = 100_000  # MCMC iterations
    K = 1000     # Number of auxiliary variables
    delta_DP = 0.001
    sigma_q_vec = [0.1, 0.01, 0.001, 0.001]
    theta_hyper = [10, 1, 0.0001, 0.01]  # [eps_var, sab, tau_var, rho_var]
    theta_init = [1, 0.5, 0, 0] # [eps_DP, s, tau, rho]
    update_params = [1, 1, 1, 1]

    # Target alpha values
    alpha_vec = np.arange(0.01, 1.0, 0.01)
    L_a = len(alpha_vec)
    n, N = 20, 100 # 20 MIAs each tested 100 times with H0 and 100 times with H1

    # Initialization
    # Using lists of lists of numpy arrays, equivalent to MATLAB's nested cell arrays
    X = [[np.zeros((N, L_a)) for _ in range(n)] for _ in range(L_f)]
    Y = [[np.zeros((N, L_a)) for _ in range(n)] for _ in range(L_f)]
    E = [[np.zeros((N, L_a)) for _ in range(n)] for _ in range(L_f)]

    Eps_DP_MCMC, S_MCMC, Tau_MCMC, Rho_MCMC = [[] for _ in range(4)]
    acf_results, lags_results = [], []

    for fn, filename in enumerate(filenames):
        print(f"\n--- Processing file: {filename} ({fn+1}/{L_f}) ---")
        print(f'../data/{filename}')
        l_0_matrix, l_1_matrix = read_losses(f'../data/{filename}')

        print("Calculating errors (Type I/II)...")
        for h in [0, 1]:  # 0 for H0 is true, 1 for H1 is true
            for k in range(n):  # Challenge base
                for j in range(N):  # Challenge instance
                    if h == 0: # H0 is true
                        l_ast = l_0_matrix[k, j]
                        l0 = np.concatenate((l_0_matrix[k, :j], l_0_matrix[k, j+1:]))
                        l1 = l_1_matrix[k, :]
                    else: # H1 is true
                        l_ast = l_1_matrix[k, j]
                        l1 = np.concatenate((l_1_matrix[k, :j], l_1_matrix[k, j+1:]))
                        l0 = l_0_matrix[k, :]
            
                    mu_l0, var_l0 = np.mean(l0), np.var(l0, ddof=1)
                    mu_l1, var_l1 = np.mean(l1), np.var(l1, ddof=1)
                    
                    D = LRT_normal(l_ast, mu_l0, var_l0, mu_l1, var_l1, alpha_vec)

                    if h == 0:
                        X[fn][k][j, :] = D  # False Positive Rate (alpha)
                    else:
                        Y[fn][k][j, :] = 1 - D # False Negative Rate (beta)
        
        for k in range(n):
            E[fn][k] = X[fn][k] + Y[fn][k]

        # Plot the error lines (alpha vs beta)
        plt.figure(figsize=(5, 4.5))
        for k in range(n):
            plt.plot(np.mean(X[fn][k], axis=0), np.mean(Y[fn][k], axis=0), '.-')
        plt.grid(True)
        plt.xlabel('$\\alpha$ (False Positive Rate)')
        plt.ylabel('$\\beta$ (False Negative Rate)')
        plt.title(f'Attack Performance for {alg_names[fn]}')
        plt.savefig(os.path.join(foldername, f'Errors_{filename}.pdf'))
        plt.close()

        # Get the (X, Y) error counts at alpha_target = 0.1
        alpha_target_idx = 9 # Corresponds to alpha = 0.1
        X0f = np.array([np.sum(X[fn][k][:, alpha_target_idx]) for k in range(n)])
        Y0f = np.array([np.sum(Y[fn][k][:, alpha_target_idx]) for k in range(n)])
        N0 = 100 * np.ones(n)
        N1 = 100 * np.ones(n)

        print(f"Running MCMC for file: {filename}...")
        start_time = time.time()
        theta_samps = MCMC_epsDP_v2(N0, N1, X0f, Y0f, M, K, delta_DP,
                                    theta_init, sigma_q_vec, theta_hyper, update_params, 'bivariate')
        print(f"MCMC finished in {time.time() - start_time:.2f} seconds.")

        Eps_DP_MCMC.append(theta_samps[0, :])
        S_MCMC.append(theta_samps[1, :])
        Tau_MCMC.append(theta_samps[2, :])
        Rho_MCMC.append(theta_samps[3, :])

        # Autocorrelation
        eps_after_burn_in = theta_samps[0, M//5:]
        acf_vals, confint = acf(eps_after_burn_in, nlags=min(len(eps_after_burn_in)-1, 5000), alpha=0.05, fft=True)
        lags = np.arange(len(acf_vals))
        acf_results.append(acf_vals)
        lags_results.append(lags)
        
        # 2D Histogram
        plt.figure(figsize=(5, 4.5))
        plt.hist2d(theta_samps[0, M//2:], theta_samps[1, M//2:], bins=50, cmap='gray', density=True)
        plt.xlabel('$\\epsilon$')
        plt.ylabel('$s$')
        plt.title(f'Posterior for {alg_names[fn]}')
        plt.colorbar(label='Frequency')
        plt.savefig(os.path.join(foldername, f'histogram_{filename}.pdf'))
        plt.close()

    # --- Final Aggregate Plots ---
    # Autocorrelation Plot
    plt.figure(figsize=(6, 4))
    for fn in range(L_f):
        plt.plot(lags_results[fn], acf_results[fn], label=alg_names[fn])
    plt.xlabel('Lag')
    plt.ylabel('Sample ACF')
    plt.xlim(0, 5000)
    plt.grid(True)
    plt.legend(fontsize='small')
    plt.savefig(os.path.join(foldername, 'auto_corr.pdf'))
    plt.close()
    
    # MCMC trace plots
    fig, axes = plt.subplots(L_f, 4, figsize=(15, 12), sharex=True)
    param_titles = ['$\\epsilon$', '$s$', '$\\tau$', '$\\rho$']
    for fn in range(L_f):
        axes[fn, 0].plot(Eps_DP_MCMC[fn])
        axes[fn, 0].set_ylabel(alg_names[fn], rotation=0, labelpad=40, va='center')
        
        axes[fn, 1].plot(S_MCMC[fn])
        axes[fn, 2].plot(Tau_MCMC[fn])
        axes[fn, 3].plot(Rho_MCMC[fn])
        
        if fn == 0:
            for j in range(4):
                axes[fn, j].set_title(param_titles[j])
        if fn == L_f - 1:
            for j in range(4):
                axes[fn, j].set_xlabel('Iteration')
    
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.savefig(os.path.join(foldername, 'MCMC_samples.pdf'))
    plt.close()

    # Save results to a file
    np.savez(os.path.join(foldername, 'real_data_experiments.npz'),
             Eps_DP_MCMC=np.array(Eps_DP_MCMC, dtype=object),
             S_MCMC=np.array(S_MCMC, dtype=object),
             Tau_MCMC=np.array(Tau_MCMC, dtype=object),
             Rho_MCMC=np.array(Rho_MCMC, dtype=object),
             acf=np.array(acf_results, dtype=object),
             lags=np.array(lags_results, dtype=object))
    
    print(f"\nAll experiments finished. Outputs are saved in '{foldername}' directory.")

if __name__ == '__main__':
    run_real_data_experiments()