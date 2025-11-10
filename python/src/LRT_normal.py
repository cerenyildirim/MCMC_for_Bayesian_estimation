import numpy as np
from scipy.stats import ncx2, norm

def LRT_normal(x, mu0, var0, mu1, var1, alpha):
    """
    Applies the decision rule of the LRT that compares two normal
    distributions with a single observation.

    Args:
        x (float or np.ndarray): The observation(s).
        mu0, var0 (float): Mean and variance of H0.
        mu1, var1 (float): Mean and variance of H1.
        alpha (float or np.ndarray): The significance level(s).

    Returns:
        bool or np.ndarray: The decision(s) (True if H0 is rejected).
    """
    if var0 == var1:
        decision = (np.sign(mu0 - mu1) * (x - mu0) / np.sqrt(var0)) > norm.ppf(1 - alpha)
    else:
        R = (mu0 / var0 - mu1 / var1) / (1 / var1 - 1 / var0)
        delta_sq = (mu0 + R)**2 / var0
        statistic = np.clip((x + R)**2 / var0, 1e-7, None)  # np.clip added to avoid zero statistic

        if var0 > var1:
            decision = ncx2.cdf(statistic, df=1, nc=delta_sq) <= alpha
        else:
            decision = ncx2.cdf(statistic, df=1, nc=delta_sq) >= (1 - alpha)
            
    return decision