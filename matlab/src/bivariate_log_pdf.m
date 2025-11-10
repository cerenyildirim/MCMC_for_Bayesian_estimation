function log_f = bivariate_log_pdf(X, Y, mu_X, mu_Y, sigma_X, sigma_Y, rho)
    % Compute the log of the bivariate normal PDF for multiple (X, Y) pairs

    % Ensure X and Y are the same size
    if ~isequal(size(X), size(Y))
        error('X and Y must have the same dimensions.');
    end
    % if sum(abs(rho(:)) > 1) > 0
    %     error('rho must be between -1 and 1');
    % end

    % Compute constants
    log_coeff = -log(2 * pi) - log(sigma_X) - log(sigma_Y) - 0.5 * log(max(0, 1 - rho.^2));

    % Compute exponent term
    exponent = (-1 ./ (2 * (1 - rho.^2))) .* ...
        ( ((X - mu_X).^2 ./ sigma_X.^2) + ((Y - mu_Y).^2 ./ sigma_Y.^2) ...
        - (2 * rho .* (X - mu_X) .* (Y - mu_Y)) ./ (sigma_X .* sigma_Y) );

    % Compute log PDF
    log_f = log_coeff + exponent;
end