import numpy as np

def bivariate_log_pdf(X, Y, mu_X, mu_Y, sigma_X, sigma_Y, rho):
    """
    Compute the log of the bivariate normal PDF for multiple (X, Y) pairs.

    Args:
        X, Y (np.ndarray): Input data arrays.
        mu_X, mu_Y (np.ndarray): Mean arrays.
        sigma_X, sigma_Y (np.ndarray): Standard deviation arrays.
        rho (np.ndarray): Correlation coefficient array.

    Returns:
        np.ndarray: The log of the bivariate normal PDF.
    """
    if not X.shape == Y.shape:
        raise ValueError('X and Y must have the same dimensions.')
    
    one_minus_rho_sq = 1 - rho**2
    
    log_coeff = -np.log(2 * np.pi) - np.log(sigma_X) - np.log(sigma_Y) - 0.5 * np.log(one_minus_rho_sq)

    x_minus_mu_x = X - mu_X
    y_minus_mu_y = Y - mu_Y

    exponent = (-1 / (2 * one_minus_rho_sq)) * (
        (x_minus_mu_x**2 / sigma_X**2) +
        (y_minus_mu_y**2 / sigma_Y**2) -
        (2 * rho * x_minus_mu_x * y_minus_mu_y) / (sigma_X * sigma_Y)
    )

    log_f = log_coeff + exponent
    return log_f