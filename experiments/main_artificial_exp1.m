% This is the main script for generating the first set of experiments in
% Section 4.1. in the paper
% "MCMC for Bayesian estimation of Differential Privacy from Membership 
% Inference Attacks" (Figure 2)

clc; clear; close all; fc = 0;
rng(1);

M = 1000000;  % MCMC iterations
K = 100; % Number of auxiliary variables
delta_DP = 0.01;
sigma_qe = 0.1; % proposal std for epsilon
log_norm_var = 10; % prior variance of log epsilon

N_vec = 100:100:1000;
N_01 = 500; N_00 = 500;
s_2 = 0.99;
L_N = length(N_vec);
s_vec = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99];
L_s = length(s_vec);

eps_DP_low = {zeros(1, L_s), zeros(1, L_N)};
eps_DP_high = {zeros(1, L_s), zeros(1, L_N)};


% The first experiment: CI vs s with fixed N
X = 200; Y = 200;

fprintf('Running the first experiment... \n');
for j = 1:L_s
    disp(j);
    s = s_vec(j);
    [eps_DP_samps] = MCMC_epsDP(N_00, N_01, X, Y, 1, delta_DP, M, K, ...
        sigma_qe, 1, log_norm_var, s, 1, 1, 0);
    eps_DP_after_burn_in = eps_DP_samps(M/10:M);
    eps_DP_low{1}(j) = quantile(eps_DP_after_burn_in, 0.05);
    eps_DP_high{1}(j) = quantile(eps_DP_after_burn_in, 0.95);    
end

% The first experiment: CI vs N with fixed s
fprintf('Running the second experiment... \n');
for i = 1:L_N
    disp(i);
    N0 = N_vec(i);
    N1 = N_vec(i);
    X = round(N0*0.4); Y = round(N1*0.4);

    [eps_DP_samps] = MCMC_epsDP(N0, N1, X, Y, 1, delta_DP, M, K, ...
        sigma_qe, 1, log_norm_var, 0.99, 1, 1, 0);
    eps_DP_after_burn_in = eps_DP_samps(M/10:M);
    eps_DP_low{2}(i) = quantile(eps_DP_after_burn_in, 0.05);
    eps_DP_high{2}(i) = quantile(eps_DP_after_burn_in, 0.95);    
end

%% CI vs s (Figure 2 Left)
fc  = fc + 1; figure;
plot(eps_DP_low{1});
hold on;
plot(eps_DP_high{1});
set(gca, 'xtick', 1:L_s, 'xticklabel', s_vec);
title('Attack sample size: 500+500, $X = Y = 200$', 'Interpreter', 'Latex');
legend('5% lower end', '95% upper end');
xlabel('$s$: strength coef. for attack performance prior', 'Interpreter', 'Latex');
ylabel('$90\%$ Cred. Int. for $\epsilon$', 'interpreter', 'latex');

%% CI vs N (Figure 2 Left)
fc  = fc + 1; figure;
plot(N_vec, eps_DP_low{2});
hold on;
plot(N_vec, eps_DP_high{2});
title('$s = 0.99$', 'Interpreter', 'Latex');
legend('5% lower end', '95% upper end');
xlabel('Sample size for attack', 'Interpreter', 'Latex');
ylabel('$90\%$ Cred. Int. for $\epsilon$', 'interpreter', 'latex');