function [eps_DP_samps, s_samps] = MCMC_epsDP(N0, N1, X, Y, eps_DP, delta_DP, M, K, sigma_qe, sigma_qs, log_norm_var, s, sa, sb, est_s)

% [eps_DP_samps, s_samps] = MCMC_epsDP(N0, N1, X, Y, eps_DP, delta_DP, M, N, sigma_q, sigma_qs, log_norm_var, s, sa, sb, est_s)
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
% eps_DP: Initial value for epsilon
% delta_DP: delta parameter of (epsilon, delta)-DP
% 
% M: Number of MCMC iterations
% K: Number of auxiliary variables (this MCMC algorithm is a MHAAR)
% 
% sigma_qe: standard deviation of the random walk proposal for epsilon
% sigma_qs: standard deviation of the random walk proposal for s
% 
% log_norm_var: The variance parameter of the log-normal prior for epsilon
%
% s: initial value for s
% sa, sb: The prior for s is Beta(sa, sb)
% est_s: Binary option. If set to 0, s is not estimated.
% 
% 
% Outputs:
% eps_DP_samps: M samples for epsilon
% s_samps: M samples s_samps

eps_DP_samps = zeros(1, M);
s_samps = zeros(1, M);

if nargin == 14
    est_s = 1;
end

% Initialize
AreaR_curr = 1 - 2 *(1 - delta_DP)^2*exp(-eps_DP)/(1 + exp(-eps_DP));
AreaR_curr_s = 1 - 2 *(1 - delta_DP*s)^2*exp(-eps_DP*s)/(1 + exp(-eps_DP*s));
n = length(X);
a = 0.5*ones(1, n);
b = 0.5*ones(1, n);

for m = 1:M
    if mod(m, 10000) == 0
        disp(m);
    end
    
    % Propose eps_DP and s
    eps_DP_prop = exp(log(eps_DP) + sigma_qe*randn);
    if est_s == 1
        s_prop = s + randn*sigma_qs;
    else
        s_prop = s;
    end

    % log prior ratio p(s', epsilon')/p(s, epsilon)
    if s == 0
        log_prior_ratio = -(eps_DP_prop^2 - eps_DP^2)/log_norm_var;
    else
        log_prior_ratio = -(eps_DP_prop^2 - eps_DP^2)/log_norm_var...
            + log(s_prop)*(sa-1) + log(1 - s_prop)*(sb-1)...
            - log(s)*(sa-1) - log(1 - s)*(sb-1);
    end

    if s_prop <= 1 && s_prop >= 0
        
        % generate samples for alpha_i and beta_i
        A = [a; rand(K-1, n)];
        B = [b; rand(K-1, n)];
    
        % Calculate the weights for the auxiliary variables
        PYX = (N0 - X).*log(1 - A) + X.*log(A) + (N1 - Y).*log(1 - B) + Y.*log(B);
    
        [InR_curr] = DPinR(A, B, eps_DP, delta_DP).*(1 - DPinR(A, B, s*eps_DP, s*delta_DP));
        [InR_prop] = DPinR(A, B, eps_DP_prop, delta_DP).*(1 - DPinR(A, B, s_prop*eps_DP_prop, s_prop*delta_DP));
    
        AreaR_prop = 1 - 2 *(1 - delta_DP)^2*exp(-eps_DP_prop)/(1 + exp(-eps_DP_prop));
        AreaR_prop_s = 1 - 2 *(1 - s*delta_DP)^2*exp(-s_prop*eps_DP_prop)/(1 + exp(-s_prop*eps_DP_prop));
    
        log_W_prop = PYX + log(double(InR_prop)) - log(AreaR_prop - AreaR_prop_s);
        log_W_curr = PYX + log(double(InR_curr)) - log(AreaR_curr - AreaR_curr_s);
    
        % acceptance rate
        max_curr = max(log_W_curr, [], 1);
        max_prop = max(log_W_prop, [], 1);
    
        log_w_numer_sum = log(sum(exp(log_W_prop - max_prop), 1)) + max_prop;
        log_w_denom_sum = log(sum(exp(log_W_curr - max_curr), 1)) + max_curr;
    
        log_numer = sum(log_w_numer_sum);
        log_denom = sum(log_w_denom_sum);
    
        log_r = log_numer - log_denom + log_prior_ratio;
    
        % decision
        if rand < exp(log_r)
            eps_DP = eps_DP_prop;
            s = s_prop;
            AreaR_curr = AreaR_prop;
            AreaR_curr_s = AreaR_prop_s;
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
    eps_DP_samps(m) = eps_DP;
    s_samps(m) = s;

end
