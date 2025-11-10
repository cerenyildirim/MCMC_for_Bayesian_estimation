import numpy as np

def DPinR(A, B, eps_DP, delta_DP):
    """
    Determines if a point in [0, 1] x [0, 1] is in the region
    R(epsilon, delta) of admissible (alpha, beta) error probabilities.

    Args:
        A (np.ndarray): Array of alpha values (Type I error).
        B (np.ndarray): Array of beta values (Type II error).
        eps_DP (float): Epsilon parameter of (epsilon, delta)-DP.
        delta_DP (float): Delta parameter of (epsilon, delta)-DP.

    Returns:
        np.ndarray: A boolean array, True for points within the region.
    """
    exp_eps = np.exp(eps_DP)
    
    Cond1 = A + exp_eps * B >= 1 - delta_DP
    Cond2 = B + exp_eps * A >= 1 - delta_DP
    Cond3 = A + exp_eps * B <= exp_eps + delta_DP 
    Cond4 = B + exp_eps * A <= exp_eps + delta_DP 

    return Cond1 & Cond2 & Cond3 & Cond4