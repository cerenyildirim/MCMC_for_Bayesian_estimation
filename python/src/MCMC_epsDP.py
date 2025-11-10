import numpy as np
from src.DPinR import DPinR

def MCMC_epsDP(N0, N1, X, Y, eps_DP, delta_DP, M, K, sigma_qe, sigma_qs, log_norm_var, s, sa, sb, est_s=1):
    """
    Implements a simplified MCMC-DP-Est algorithm.

    Args:
        N0, N1 (np.ndarray): Vectors of challenge numbers.
        X, Y (np.ndarray): Vectors of false positives and false negatives.
        eps_DP (float): Initial value for epsilon.
        delta_DP (float): Delta parameter of (epsilon, delta)-DP.
        M (int): Number of MCMC iterations.
        K (int): Number of auxiliary variables.
        sigma_qe, sigma_qs (float): Proposal standard deviations.
        log_norm_var (float): Variance for the log-normal prior of epsilon.
        s (float): Initial value for s.
        sa, sb (float): Shape parameters for the Beta prior of s.
        est_s (int): If 0, s is not estimated.

    Returns:
        tuple[np.ndarray, np.ndarray]: M samples for epsilon and s.
    """
    eps_DP_samps = np.zeros(M)
    s_samps = np.zeros(M)

    # Initialize
    AreaR_curr = 1 - 2 * (1 - delta_DP)**2 * np.exp(-eps_DP) / (1 + np.exp(-eps_DP))
    AreaR_curr_s = 1 - 2 * (1 - delta_DP * s)**2 * np.exp(-eps_DP * s) / (1 + np.exp(-eps_DP * s))
    n = len(X)
    a = 0.5 * np.ones(n)
    b = 0.5 * np.ones(n)

    for m in range(M):
        if (m + 1) % 10000 == 0:
            print(m + 1)
        
        # Propose eps_DP and s
        eps_DP_prop = np.exp(np.log(eps_DP) + sigma_qe * np.random.randn())
        s_prop = s + np.random.randn() * sigma_qs if est_s == 1 else s

        if 0 <= s_prop <= 1:
            # Log prior ratio
            if s == 0:
                log_prior_ratio = -(eps_DP_prop**2 - eps_DP**2) / log_norm_var
            else:
                log_prior_ratio = (-(eps_DP_prop**2 - eps_DP**2) / log_norm_var
                                   + (sa - 1) * np.log(s_prop) + (sb - 1) * np.log(1 - s_prop)
                                   - (sa - 1) * np.log(s) - (sb - 1) * np.log(1 - s))

            # Generate auxiliary variables
            A = np.vstack([a, np.random.rand(K - 1, n)])
            B = np.vstack([b, np.random.rand(K - 1, n)])

            # Calculate weights
            PYX = (N0 - X) * np.log(1 - A) + X * np.log(A) + (N1 - Y) * np.log(1 - B) + Y * np.log(B)

            InR_curr = DPinR(A, B, eps_DP, delta_DP) * (1 - DPinR(A, B, s * eps_DP, s * delta_DP))
            InR_prop = DPinR(A, B, eps_DP_prop, delta_DP) * (1 - DPinR(A, B, s_prop * eps_DP_prop, s_prop * delta_DP))
            
            AreaR_prop = 1 - 2 * (1 - delta_DP)**2 * np.exp(-eps_DP_prop) / (1 + np.exp(-eps_DP_prop))
            AreaR_prop_s = 1 - 2 * (1 - s_prop * delta_DP)**2 * np.exp(-s_prop * eps_DP_prop) / (1 + np.exp(-s_prop * eps_DP_prop))

            # Add a small constant to avoid log(0)
            log_W_prop = PYX + np.log(InR_prop.astype(float) + 1e-100) - np.log(AreaR_prop - AreaR_prop_s + 1e-100)
            log_W_curr = PYX + np.log(InR_curr.astype(float) + 1e-100) - np.log(AreaR_curr - AreaR_curr_s + 1e-100)

            # Acceptance rate (with log-sum-exp trick)
            max_curr = np.max(log_W_curr, axis=0)
            max_prop = np.max(log_W_prop, axis=0)
            log_w_numer_sum = np.log(np.sum(np.exp(log_W_prop - max_prop), axis=0)) + max_prop
            log_w_denom_sum = np.log(np.sum(np.exp(log_W_curr - max_curr), axis=0)) + max_curr
            log_r = np.sum(log_w_numer_sum) - np.sum(log_w_denom_sum) + log_prior_ratio
            
            if np.random.rand() < np.exp(log_r):
                eps_DP, s = eps_DP_prop, s_prop
                AreaR_curr, AreaR_curr_s = AreaR_prop, AreaR_prop_s
                W_temp = np.exp(log_W_prop - log_w_numer_sum)
            else:
                W_temp = np.exp(log_W_curr - log_w_denom_sum)

            # Sample (alpha, beta) pairs
            rand_uniform = np.random.rand(1, n)
            temp_ind_vec = np.sum(rand_uniform > np.cumsum(W_temp, axis=0), axis=0)
            a = A[temp_ind_vec, np.arange(n)]
            b = B[temp_ind_vec, np.arange(n)]

        # Store the sampled values
        eps_DP_samps[m] = eps_DP
        s_samps[m] = s

    return eps_DP_samps, s_samps