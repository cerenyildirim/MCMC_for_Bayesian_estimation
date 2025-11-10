import numpy as np

def gen_beta_DP(eps_DP, delta_DP):
    """
    Generates the boundary of the privacy region R for a given (epsilon, delta).
    This function defines the y-coordinates for specific x-coordinates on the boundary.

    Args:
        eps_DP (float): The epsilon value for differential privacy.
        delta_DP (float): The delta value for differential privacy.

    Returns:
        tuple[np.ndarray, np.ndarray]: x and y coordinates of the boundary points.
    """
    c = (1 - delta_DP) / (1 + np.exp(eps_DP))

    x = np.array([0, c, 1 - delta_DP, 1])
    y = np.zeros_like(x, dtype=float)

    cond1 = x < c
    cond2 = (x >= c) & (x < (1 - delta_DP))
    # cond3 (x >= 1 - delta_DP) is implicitly handled by initialization

    y[cond1] = 1 - delta_DP - np.exp(eps_DP) * x[cond1]
    y[cond2] = c - np.exp(-eps_DP) * (x[cond2] - c)
    # y[cond3] remains 0 as initialized
    
    return x, y