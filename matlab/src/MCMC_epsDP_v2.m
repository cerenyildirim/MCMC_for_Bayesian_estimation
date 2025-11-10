function [theta_samps] = MCMC_epsDP_v2(N0, N1, X, Y, M, K, delta_DP, theta, sigma_q_vec, theta_hyper, update_params, g_model)

% [theta_samps] = MCMC_epsDP_v2(N0, N1, X, Y, M, K, delta_DP, theta, sigma_q_vec, theta_hyper, update_params, g_model)
%
% 
% This function implements MCMC-DP-Est in Algoritm 1 of the paper
% "MCMC for Bayesian estimation of Differential Privacy from Membership 
% Inference Attacks"
% 
% Inputs:
% N0: (1 x n) vector of numbers of challenges with theta ~ H0
% N1: (1 x n) vector of numbers of challenges with theta ~ H1
% 
% X: (1 x n) vector of false positives out of the N0 tests where H0 is true
% Y: (1 x n) vector of false negatives out of the N0 tests where H0 is true
% 
% delta_DP: delta parameter of (epsilon, delta)-DP
% 
% M: Number of MCMC iterations
% K: Number of auxiliary variables (this MCMC algorithm is a MHAAR)
%
% theta: Initial value for (eps_DP, s, tau, rho)
%
% theta_hyper: hyperparameters for the prior of the components in theta
% 
% Outputs:
% theta_samps: M samples for [epsilon, s, tau, rho]

if nargin == 11
    g_model = 'Binomial';
end

if strcmp(g_model, 'Binomial') == 1
    update_params(3:4) = 0;
end

% Initialize
theta_samps = zeros(4, M);
eps_DP = theta(1);
s = theta(2);
tau = theta(3);
rho = theta(4);

if update_params(3) == 1
    tau = 0;
end
if update_params(4) == 1
    rho = 0;
end

AreaR_curr = 1 - 2 *(1 - delta_DP)^2*exp(-eps_DP)/(1 + exp(-eps_DP));
AreaR_curr_s = 1 - 2 *(1 - delta_DP*s)^2*exp(-eps_DP*s)/(1 + exp(-eps_DP*s));

min_bound_tau = min( min(1./(N0-1)), min(1./(N1-1)));
min_bound_rho_curr = min(sqrt(N0.*N1)./sqrt((1 + (N0 -1)*tau).*(1 + (N1 -1)*tau)));

n = length(X);
a = 0.5*ones(1, n);
b = 0.5*ones(1, n);

for m = 1:M
    if mod(m, 1000) == 0
        disp(m);
    end
    
    % Propose eps_DP and s
    eps_DP_prop = exp(log(eps_DP) + update_params(1)*sigma_q_vec(1)*randn);
    s_prop = s + update_params(2)*randn*sigma_q_vec(2);
    tau_prop = tau + update_params(3)*randn*sigma_q_vec(3);
    rho_prop = rho + update_params(4)*randn*sigma_q_vec(4);

    log_q_ratio = log(eps_DP_prop) - log(eps_DP);
    
    min_bound_rho_prop = min(sqrt((1 + (N0 -1)*tau_prop).*(1 + (N1 -1)*tau_prop))./sqrt(N0.*N1));

    if abs(s_prop - 0.5) <= 0.5 && tau_prop > -min_bound_tau && tau_prop < 1 && abs(rho_prop) < min_bound_rho_prop

        % log prior ratio p(s', epsilon')/p(s, epsilon)
        if s == 0
            log_prior_ratio = -0.5*(eps_DP_prop^2 - eps_DP^2)/theta_hyper(1);
        else
            log_prior_ratio = -0.5*(eps_DP_prop^2 - eps_DP^2)/theta_hyper(1)...
                - log(2*min_bound_rho_prop) + log(2*min_bound_rho_curr)...
                + log(s_prop)*(theta_hyper(2)-1) ...
                + log(1 - s_prop)*(theta_hyper(2)-1)...
                - log(s)*(theta_hyper(2)-1) ...
                - log(1 - s)*(theta_hyper(2)-1)...
                - 0.5*(tau_prop^2 - tau^2)/theta_hyper(3);
        end
        
        % generate samples for alpha_i and beta_i
        A = [a; rand(K-1, n)];
        B = [b; rand(K-1, n)];
        
        % Calculate the weights for the auxiliary variables
        if strcmp(g_model, 'bivariate') == 1
            sigma_X_curr = sqrt(N0.*A.*(1-A) + tau*N0.*(N0-1).*A.*(1-A));
            sigma_Y_curr = sqrt(N1.*B.*(1-B) + tau*N1.*(N1-1).*B.*(1-B));
            sigma_X_prop = sqrt(N0.*A.*(1-A) + tau_prop*N0.*(N0-1).*A.*(1-A));
            sigma_Y_prop = sqrt(N1.*B.*(1-B) + tau_prop*N1.*(N1-1).*B.*(1-B));
            Cov_XY_curr = rho*sqrt(A.*(1-A).*B.*(1-B)).*N0.*N1;
            Cov_XY_prop = rho_prop*sqrt(A.*(1-A).*B.*(1-B)).*N0.*N1;
    
            corr_XY_curr = Cov_XY_curr./(sigma_X_curr.*sigma_Y_curr);
            corr_XY_prop = Cov_XY_prop./(sigma_X_prop.*sigma_Y_prop);
             
            mu_X = N0.*A;
            mu_Y = N1.*B;
    
            PYX = bivariate_log_pdf(X, Y, mu_X, mu_Y, sigma_X_curr, sigma_Y_curr, corr_XY_curr);
            PYX_prop = bivariate_log_pdf(X, Y, mu_X, mu_Y, sigma_X_prop, sigma_Y_prop, corr_XY_prop);

        elseif strcmp(g_model, 'Binomial') == 1
            PYX = (N0 - X).*log(1 - A) + X.*log(A) + (N1 - Y).*log(1 - B) + Y.*log(B);
            PYX_prop = PYX;
        end

        [InR_curr] = DPinR(A, B, eps_DP, delta_DP).*(1 - DPinR(A, B, s*eps_DP, s*delta_DP));
        [InR_prop] = DPinR(A, B, eps_DP_prop, delta_DP).*(1 - DPinR(A, B, s_prop*eps_DP_prop, s_prop*delta_DP));
    
        AreaR_prop = 1 - 2 *(1 - delta_DP)^2*exp(-eps_DP_prop)/(1 + exp(-eps_DP_prop));
        AreaR_prop_s = 1 - 2 *(1 - s_prop*delta_DP)^2*exp(-s_prop*eps_DP_prop)/(1 + exp(-s_prop*eps_DP_prop));
    
        log_W_prop = PYX_prop + log(double(InR_prop)) - log(AreaR_prop - AreaR_prop_s);
        log_W_curr = PYX + log(double(InR_curr)) - log(AreaR_curr - AreaR_curr_s);
    
        % acceptance rate
        max_curr = max(log_W_curr, [], 1);
        max_prop = max(log_W_prop, [], 1);
    
        log_w_numer_sum = log(sum(exp(log_W_prop - max_prop), 1)) + max_prop;
        log_w_denom_sum = log(sum(exp(log_W_curr - max_curr), 1)) + max_curr;
    
        log_numer = sum(log_w_numer_sum);
        log_denom = sum(log_w_denom_sum);
    
        log_r = log_numer - log_denom + log_prior_ratio + log_q_ratio;
    
        % decision
        if rand < exp(log_r)
            eps_DP = eps_DP_prop;
            s = s_prop;
            tau = tau_prop;
            rho = rho_prop;
            AreaR_curr = AreaR_prop;
            AreaR_curr_s = AreaR_prop_s;
            min_bound_rho_curr = min_bound_rho_prop;
            W_temp = exp(log_W_prop - log_w_numer_sum);
        else
            W_temp = exp(log_W_curr - log_w_denom_sum);
        end
    
        % sample the (alpha, beta) pairs
        temp_ind_vec = sum(rand(1, n) > cumsum(W_temp)) + 1;
        temp_mtx_ind_vec = K*(0:n-1) + temp_ind_vec;
        a = A(temp_mtx_ind_vec);
        b = B(temp_mtx_ind_vec);
    end

    % Store the sampled values
    theta_samps(:, m) = [eps_DP, s, tau, rho]';
end

