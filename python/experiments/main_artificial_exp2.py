# main_artificial_exp2.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time
from src.MCMC_epsDP import MCMC_epsDP
from src.gen_beta_DP import gen_beta_DP

def run_experiment_2():
    """
    Main script for generating the second set of experiments in Section 4.1.
    (Corresponds to Figure 3 in the paper)
    """
    print("Starting experiment: Joint Posterior and Privacy Regions")
    # Set up matplotlib for LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    
    # --- Generate Artificial Error Counts ---
    np.random.seed(1)
    weak_or_strong = 1 # 0: weak, 1: strong

    if weak_or_strong == 0:
        # Weak attacker
        a, b, n = 10, 10, 10
        pa_vec = np.random.beta(a, b, size=n)
        pb_vec = np.random.beta(a, b, size=n)
        N0 = 1000 * np.ones(n)
        N1 = 1000 * np.ones(n)
        X = np.random.binomial(N0, pa_vec)
        Y = np.random.binomial(N1, pb_vec)
    else:
        # Strong attacker
        n = 10
        N0 = 10 * np.array([100] * n)
        N1 = 10 * np.array([100] * n)
        X = 10 * np.array([10, 20, 11, 12, 20, 5, 6, 4, 20, 10])
        Y = 10 * np.array([10, 8, 10, 10, 6, 20, 15, 25, 7, 12])

    # --- Run MCMC ---
    print("Running MCMC...")
    start_time = time.time()
    M = 1_000_000 # MCMC iterations (reduced for faster execution, original was 1M)
    K = 1000       # Number of auxiliary variables
    delta_DP = 0.01
    eps_DP0, s0 = 10.0, 0.5  # Initial values
    
    eps_DP_samps, s_samps = MCMC_epsDP(
        N0, N1, X, Y, eps_DP0, delta_DP, M, K,
        sigma_qe=0.1, sigma_qs=0.01, log_norm_var=10,
        s=s0, sa=1, sb=1, est_s=1
    )
    print(f"MCMC finished in {time.time() - start_time:.2f} seconds.")

    burn_in_idx = M // 2
    eps_after_burn = eps_DP_samps[burn_in_idx:]
    s_after_burn = s_samps[burn_in_idx:]

    # --- Plotting ---
    print("Generating plots...")
    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(1, 4)

    # Subplot 1: Privacy Regions
    ax1 = fig.add_subplot(gs[0, 0])
    MH = 50 # Number of bins for histogram
    counts, bin_edges = np.histogram(eps_after_burn, bins=MH, density=True)
    
    A_prev, B_prev = None, None
    for i in range(MH):
        eps_val = bin_edges[i+1]
        A, B = gen_beta_DP(eps_val, delta_DP)
        if i > 0:
            verts = np.vstack([np.column_stack([A, B]), np.column_stack([A_prev[::-1], B_prev[::-1]])])
            color_val = 1 - 0.7 * min(1, counts[i] / np.max(counts))
            poly = Polygon(verts, facecolor=str(color_val), edgecolor='none')
            ax1.add_patch(poly)
        A_prev, B_prev = A, B
        
    ax1.plot(X / N0, Y / N1, 'r.', markersize=8, label='Attack Perf. $(\\hat{\\alpha}_i, \\hat{\\beta}_i)$')
    ax1.set_xlabel('$\\alpha$ (Type I Error)')
    ax1.set_ylabel('$\\beta$ (Type II Error)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()

    # Subplot 2: Joint Posterior Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist2d(eps_after_burn, s_after_burn, bins=50, cmap='gray', density=True)
    ax2.set_xlabel('$\\epsilon$')
    ax2.set_ylabel('$s$')

    # Subplot 3: Marginal for epsilon
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(eps_after_burn, bins=MH, density=True, color='lightgray', edgecolor='black')
    ax3.set_xlabel('$\\epsilon$')
    ax3.set_ylabel('Posterior Density')

    # Subplot 4: Marginal for s
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(s_after_burn, bins=MH, density=True, color='lightgray', edgecolor='black')
    ax4.set_xlabel('$s$')
    ax4.set_ylabel('Posterior Density')
    
    plt.tight_layout()
    plt.savefig("figure3_artificial_exp2.pdf")
    plt.show()

if __name__ == '__main__':
    run_experiment_2()