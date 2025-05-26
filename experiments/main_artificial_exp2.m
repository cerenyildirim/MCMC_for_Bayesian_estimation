% This is the main script for generating the second set of experiments in
% Section 4.1. in the paper (Figure 3)
% "MCMC for Bayesian estimation of Differential Privacy from Membership 
% Inference Attacks"

clc; clear; close all; fc = 0;
rng(1);

%% Generate artificial error counts
weak_or_strong = 1; % 0: weak, 1: strong

if weak_or_strong == 0
    a = 10;
    b = 10;
    n = 10;
    pa_vec = betarnd(a, b, 1, n);
    pb_vec = betarnd(a, b, 1, n);
    N0 = 1000*ones(1, n);
    N1 = 1000*ones(1, n);
    X = binornd(N0, pa_vec);
    Y = binornd(N1, pb_vec);
else
    N0 = 10*[100 100 100 100 100 100 100 100 100 100]; 
    N1 = 10*[100 100 100 100 100 100 100 100 100 100];
    X = 10*[10 20 11, 12, 20, 5, 6, 4, 20, 10]; 
    Y = 10*[10 8, 10, 10, 6, 20, 15, 25, 7, 12];
end

%% Run MCMC
M = 1000000; % MCMC iterations
K = 1000; % Number of auxiliary variables
delta_DP = 0.01;
sigma_qe = 0.1; % proposal std for epsilon
sigma_qs = 0.01; % proposal std for s
log_norm_var = 10; % prior variance of log epsilon
sa = 1; sb = 1; % The prior for s is Beta(sa, sb)
eps_DP0 = 10; % initial value for epsilon
s0 = 0.5; % initial value for s

[eps_DP_samps, s_samps] = MCMC_epsDP(N0, N1, X, Y, eps_DP0, delta_DP, M,...
    K, sigma_qe, sigma_qs, log_norm_var, s0, sa, sb);
eps_DP_after_burn_in = eps_DP_samps(end/2:end);
eps_DP_low = quantile(eps_DP_after_burn_in, 0.05);
eps_DP_high = quantile(eps_DP_after_burn_in, 0.95);    

%% Plot the results
MH = 50;
fc = fc + 1; figure(fc);

subplot(1, 4, 2);
histogram2(eps_DP_samps(end/2:end), s_samps(end/2:end), 'DisplayStyle','tile','ShowEmptyBins','on');
colormap(gray);
xlabel('$\epsilon$', 'Interpreter', 'Latex');
ylabel('$s$', 'Interpreter', 'Latex');

subplot(1, 4, 3);
[H] = histogram(eps_DP_samps(end/2:end), MH, 'Normalization', 'pdf');
D = H.BinEdges;
D_mid = (D(1:end-1) + D(2:end))/2;
xlabel('$\epsilon$', 'Interpreter', 'Latex');
Values = H.BinCounts;

subplot(1, 4, 4);
[H] = histogram(s_samps(end/2:end), MH, 'Normalization', 'pdf');
xlabel('$s$', 'Interpreter', 'Latex');

subplot(1, 4, 1);
for i = 1:MH
    [A, B] = gen_beta_DP(D(i+1), delta_DP);
    if i > 1
        patch([A flip(A_prev)]', [B flip(B_prev)]', 1 - 0.5*min(1, [1 1 1]*10*(Values(i)/M)), 'EdgeColor','none')
    end
    hold on;
    B_prev = B;
    A_prev = A;
end
xlabel('$\alpha$', 'Interpreter', 'Latex');
ylabel('$\beta$', 'Interpreter', 'Latex');
set(gca, 'xlim', [-0, 1]);
set(gca, 'ylim', [-0, 1]);
hold on;
plot(X./N0, Y./N1, '.r');
hold off;
grid on;
