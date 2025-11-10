import numpy as np
from src.DPinR import DPinR
from src.bivariate_log_pdf import bivariate_log_pdf
import scipy

def MCMC_epsDP_v2(N0, N1, X, Y, M, K, delta_DP, theta, sigma_q_vec, theta_hyper, update_params, g_model='Binomial'):
    """
    Implements the MCMC-DP-Est algorithm.

    Args:
        N0, N1 (np.ndarray): Vectors of challenge numbers.
        X, Y (np.ndarray): Vectors of false positives and false negatives.
        M (int): Number of MCMC iterations.
        K (int): Number of auxiliary variables.
        delta_DP (float): Delta parameter of (epsilon, delta)-DP.
        theta (list/np.ndarray): Initial values for [eps_DP, s, tau, rho].
        sigma_q_vec (list/np.ndarray): Proposal standard deviations.
        theta_hyper (list/np.ndarray): Hyperparameters for the prior.
        update_params (list/np.ndarray): Flags to update parameters.
        g_model (str): The generative model ('Binomial' or 'bivariate').

    Returns:
        np.ndarray: M samples for [epsilon, s, tau, rho].
    """
    if g_model == 'Binomial':
        update_params[2:4] = [0, 0]

    # Initialize
    theta_samps = np.zeros((4, M))
    eps_DP, s, tau, rho = theta

    if update_params[2] == 1:
        tau = 0
    if update_params[3] == 1:
        rho = 0

    AreaR_curr = 1 - 2 * (1 - delta_DP)**2 * np.exp(-eps_DP) / (1 + np.exp(-eps_DP))
    AreaR_curr_s = 1 - 2 * (1 - delta_DP * s)**2 * np.exp(-eps_DP * s) / (1 + np.exp(-eps_DP * s))

    min_bound_tau = min(np.min(1. / (N0 - 1)), np.min(1. / (N1 - 1)))
    min_bound_rho_curr = np.min(np.sqrt(N0 * N1) / np.sqrt((1 + (N0 - 1) * tau) * (1 + (N1 - 1) * tau)))

    n = len(X)
    a = 0.5 * np.ones(n)
    b = 0.5 * np.ones(n)

    for m in range(M):
        if (m + 1) % 1000 == 0:
            print(m + 1)

        eps_DP_prop = np.exp(np.log(eps_DP) + update_params[0] * sigma_q_vec[0] * np.random.randn())
        s_prop = s + update_params[1] * np.random.randn() * sigma_q_vec[1]
        tau_prop = tau + update_params[2] * np.random.randn() * sigma_q_vec[2]
        rho_prop = rho + update_params[3] * np.random.randn() * sigma_q_vec[3]

        log_q_ratio = np.log(eps_DP_prop) - np.log(eps_DP)
        print(log_q_ratio)
        
        min_bound_rho_prop = np.min(np.sqrt((1 + (N0 - 1) * tau_prop) * (1 + (N1 - 1) * tau_prop)) / np.sqrt(N0 * N1))

        if abs(s_prop - 0.5) <= 0.5 and -min_bound_tau < tau_prop < 1 and abs(rho_prop) < min_bound_rho_prop:
            if s == 0:
                log_prior_ratio = -0.5 * (eps_DP_prop**2 - eps_DP**2) / theta_hyper[0]
            else:
                log_prior_ratio = (-0.5 * (eps_DP_prop**2 - eps_DP**2) / theta_hyper[0]
                                   - np.log(2 * min_bound_rho_prop) + np.log(2 * min_bound_rho_curr)
                                   + (theta_hyper[1] - 1) * (np.log(s_prop) + np.log(1 - s_prop))
                                   - (theta_hyper[1] - 1) * (np.log(s) + np.log(1 - s))
                                   - 0.5 * (tau_prop**2 - tau**2) / theta_hyper[2])

            A = np.vstack([a, np.random.rand(K - 1, n)])
            B = np.vstack([b, np.random.rand(K - 1, n)])

            if g_model == 'bivariate':
                sigma_X_curr = np.sqrt(N0 * A * (1 - A) + tau * N0 * (N0 - 1) * A * (1 - A))
                sigma_Y_curr = np.sqrt(N1 * B * (1 - B) + tau * N1 * (N1 - 1) * B * (1 - B))
                sigma_X_prop = np.sqrt(N0 * A * (1 - A) + tau_prop * N0 * (N0 - 1) * A * (1 - A))
                sigma_Y_prop = np.sqrt(N1 * B * (1 - B) + tau_prop * N1 * (N1 - 1) * B * (1 - B))
                Cov_XY_curr = rho * np.sqrt(A * (1 - A) * B * (1 - B)) * N0 * N1
                Cov_XY_prop = rho_prop * np.sqrt(A * (1 - A) * B * (1 - B)) * N0 * N1
                corr_XY_curr = Cov_XY_curr / (sigma_X_curr * sigma_Y_curr)
                corr_XY_prop = Cov_XY_prop / (sigma_X_prop * sigma_Y_prop)
                mu_X, mu_Y = N0 * A, N1 * B
                PYX = bivariate_log_pdf(X, Y, mu_X, mu_Y, sigma_X_curr, sigma_Y_curr, corr_XY_curr)
                PYX_prop = bivariate_log_pdf(X, Y, mu_X, mu_Y, sigma_X_prop, sigma_Y_prop, corr_XY_prop)
            elif g_model == 'Binomial':
                PYX = (N0 - X) * np.log(1 - A) + X * np.log(A) + (N1 - Y) * np.log(1 - B) + Y * np.log(B)
                PYX_prop = PYX

            InR_curr = DPinR(A, B, eps_DP, delta_DP) * (1 - DPinR(A, B, s * eps_DP, s * delta_DP))
            InR_prop = DPinR(A, B, eps_DP_prop, delta_DP) * (1 - DPinR(A, B, s_prop * eps_DP_prop, s_prop * delta_DP))

            AreaR_prop = 1 - 2 * (1 - delta_DP)**2 * np.exp(-eps_DP_prop) / (1 + np.exp(-eps_DP_prop))
            AreaR_prop_s = 1 - 2 * (1 - s_prop * delta_DP)**2 * np.exp(-s_prop * eps_DP_prop) / (1 + np.exp(-s_prop * eps_DP_prop))

            # Add a small epsilon to the denominator to avoid division by zero
            safe_denom_curr = AreaR_curr - AreaR_curr_s
            safe_denom_prop = AreaR_prop - AreaR_prop_s
            
            log_W_prop = PYX_prop + np.log(InR_prop.astype(float)) - np.log(safe_denom_prop)# if safe_denom_prop != 0 else 1e-9) #!
            log_W_curr = PYX + np.log(InR_curr.astype(float)) - np.log(safe_denom_curr)# if safe_denom_curr != 0 else 1e-9) #!

            max_curr = np.max(log_W_curr, axis=0)
            max_prop = np.max(log_W_prop, axis=0)
            log_w_numer_sum = scipy.special.logsumexp(log_W_prop - max_prop, axis=0) + max_prop #!
            log_w_denom_sum = scipy.special.logsumexp(log_W_curr - max_curr, axis=0) + max_curr #!
            log_numer = np.sum(log_w_numer_sum)
            log_denom = np.sum(log_w_denom_sum)
            log_r = log_numer - log_denom + log_prior_ratio + log_q_ratio

            if np.random.rand() < np.exp(log_r):
                eps_DP, s, tau, rho = eps_DP_prop, s_prop, tau_prop, rho_prop
                AreaR_curr, AreaR_curr_s = AreaR_prop, AreaR_prop_s
                min_bound_rho_curr = min_bound_rho_prop
                W_temp = np.exp(log_W_prop - log_w_numer_sum)
            else:
                W_temp = np.exp(log_W_curr - log_w_denom_sum)

            rand_uniform = np.random.rand(1, n)
            temp_ind_vec = np.sum(rand_uniform > np.cumsum(W_temp, axis=0), axis=0)
            a = A[temp_ind_vec, np.arange(n)]
            b = B[temp_ind_vec, np.arange(n)]

        theta_samps[:, m] = [eps_DP, s, tau, rho]
    
    return theta_samps