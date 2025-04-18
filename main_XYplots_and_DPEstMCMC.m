% This is the script that generates Figures 4, 5, 6 and Table 1 in the
% paper "MCMC for Bayesian estimation of Differential Privacy from
% Membership Inference Attacks"

% Read the file
clear all; clc; close all; fc = 0;
rng(1);

%% Setup and initizalization
foldername = 'Output_Figures';
if ~exist(foldername, 'dir')
    mkdir(foldername);
end

% Files where the loss metrics are stored
filenames = {
    'dp_attack_wo_DP_fixed_weights_0.1.txt', ...
    'dp_attack_wo_DP_random_weights_0.1.txt', ...
    'dp_attack_w_DP_fixed_weights_0.1.txt', ...
    'dp_attack_w_DP_random_weights_0.1.txt', ...
    'dp_attack_w_DP_random_weights_0.05.txt', ...
    'dp_attack_w_DP_random_weights_0.01.txt'
    };

alg_names = {'$\mathcal{A}_{1}$', '$\mathcal{A}_{2}$',...
    '$\mathcal{A}_{3}$, $\sigma = 0.1$', ...
    '$\mathcal{A}_{4}$, $\sigma = 0.1$',...
    '$\mathcal{A}_{4}$, $\sigma = 0.05$', ...
    '$\mathcal{A}_{4}$, $\sigma = 0.01$'};

L_f = length(filenames);

%% MCMC parameters
M = 100000;  % MCMC iterations
K = 1000; % Number of auxiliary variables
delta_DP = 0.001;

sigma_q_vec = [0.1, 0.01, 0.001, 0.001];

% hyperparameters: [eps_var, sab, tau_var, rho_var];
theta_hyper = [10, 1, 0.0001, 0.01];

% initial value for [eps_DP, s, tau, rho]
theta = [1, 0.5, 0, 0];

update_params = [1, 1, 1, 1];

% target alpha values
alpha_vec = 0.01:0.01:0.99;
L_a = length(alpha_vec);

% 20 MIAs each tested 100 times with H0 and 100 times with H1
n = 20; N = 100;

% Initialization
eps_DP_low = zeros(1, L_f);
eps_DP_high = zeros(1, L_f);
s_low = zeros(1, L_f);
s_high = zeros(1, L_f);

X = repmat({repmat({zeros(N, L_a)}, 1, n)}, 1, L_f);
Y = repmat({repmat({zeros(N, L_a)}, 1, n)}, 1, L_f);
E = repmat({repmat({zeros(N, L_a)}, 1, n)}, 1, L_f);


Eps_DP_MCMC = cell(1, L_f);
S_MCMC = cell(1, L_f);
Tau_MCMC = cell(1, L_f);
Rho_MCMC = cell(1, L_f);

acf = cell(1, L_f);
lags = cell(1, L_f);

for fn = 1:L_f
    filename = filenames{fn};

    [l_0_matrix, l_1_matrix] = read_losses(filename);

    fprintf('Calculating errors for file fn: %d \n', fn);
    % Find x's
    for h = [0, 1]
        for k = 1:n % challenge base
            for j = 1:N % challange
                if h == 0
                    % This point is challenged
                    l_ast = l_0_matrix(k, j);
                    % The rest are used to learn H0 and H1
                    l0 = l_0_matrix(k, [1:(j-1), (j+1):end]);
                    l1 = l_1_matrix(k, :);
                else
                    % This point is challenged
                    l_ast = l_1_matrix(k, j);
                    % The rest are used to learn H0 and H1
                    l1 = l_1_matrix(k, [1:(j-1), (j+1):end]);
                    l0 = l_0_matrix(k, :);
                end
                % Learn H0 and H1
                mu_l0 = mean(l0); mu_l1 = mean(l1);
                var_ell0 = var(l0); var_ell1 = var(l1);

                % Decisions of LRT with the fitted H0 and H1
                D = LRT_normal(l_ast, mu_l0, var_ell0, mu_l1, var_ell1, alpha_vec);

                if h == 0 % H0 is true
                    X{fn}{k}(j, :) = D;
                else % H1 is true
                    Y{fn}{k}(j, :) = 1-D;
                end
            end
        end
    end
    % Total error counts
    for k = 1:n
        E{fn}{k} = X{fn}{k} + Y{fn}{k};
    end

    %% Plot the error lines
    fc = fc + 1;
    f = figure(fc);
    set(gcf, 'Visible', 'off'); % Hide it
    f.Position = [100 100 200 170];
    set(gcf, 'renderer', 'painters');

    for k = 1:n
        plot(mean(X{fn}{k}), mean(Y{fn}{k}), '.-');
        hold on;
    end
    hold off;
    grid on;

    exportgraphics(gcf,[foldername, sprintf('/Errors_%s.pdf', filename)]);
    close(f);

    %% Run the MCMC algorithm to estimate epsilon and s
    X0f = zeros(1, n);
    Y0f = zeros(1, n);
    N0 = 100*ones(1, n);
    N1 = 100*ones(1, n);

    % Get the (X, Y) values at alpha_t = 0.1
    for k = 1:n
        X0f(k) = sum(X{fn}{k}(:, 10));
        Y0f(k) = sum(Y{fn}{k}(:, 10));
    end

    fprintf('Running MCMC for fn: %d \n', fn);

    tic;
    [theta_samps] = MCMC_epsDP_v2(N0, N1, X0f, Y0f, M, K, delta_DP, ...
        theta, sigma_q_vec, theta_hyper, update_params, 'bivariate');

    eps_DP_samps = theta_samps(1, :);
    s_samps = theta_samps(2, :);
    tau_samps = theta_samps(3, :);
    rho_samps = theta_samps(4, :);

    toc;
    % Store the estimates epsilon and s
    Eps_DP_MCMC{fn} = eps_DP_samps;
    S_MCMC{fn} = s_samps;
    Tau_MCMC{fn} = tau_samps;
    Rho_MCMC{fn} = rho_samps;

    eps_DP_after_burn_in = eps_DP_samps(M/5:M);
    
    [acf{fn}, lags{fn}] = autocorr(eps_DP_after_burn_in, 'Numlags', min(M-1, 5000));

    % Get the credible intervals for epsilon and s
    eps_DP_low(fn) = quantile(eps_DP_after_burn_in, 0.05);
    eps_DP_high(fn) = quantile(eps_DP_after_burn_in, 0.95);

    s_after_burn_in = s_samps(M/10:M);
    s_low(fn) = quantile(s_after_burn_in, 0.05);
    s_high(fn) = quantile(s_after_burn_in, 0.95);

end

%%
fc = fc + 1;
f = figure(fc);
set(gcf, 'Visible', 'off'); % Hide it
f.Position = [100 100 200 100];
for fn = 1:L_f
    set(gcf, 'renderer', 'painters');
    plot(lags{fn}, acf{fn});
    hold on;
end
xlabel('lag'); ylabel('Sample ACF');
set(gca, 'xlim', [0, 5000]);

grid on;
exportgraphics(gcf,[foldername '/auto_corr.pdf']);
close(f);

save('real_data_experiments');



%% plot the samples

f = figure(fc);
f.Position = [100 100 800 500];

for fn = 1:L_f

    subplot(6, 4, (fn-1)*4 + 1);
    plot(Eps_DP_MCMC{fn});
    ylabel(alg_names{fn}, 'Interpreter', 'Latex');
    if fn < L_f
        set(gca, 'xticklabel', []);
    else
        xlabel('iteration');
    end
    title('$\epsilon$', 'Interpreter', 'Latex');
    subplot(6, 4, (fn-1)*4 + 2);
    plot(S_MCMC{fn});
    title('$s$', 'Interpreter', 'Latex');
    if fn < L_f
        set(gca, 'xticklabel', []);
    else
        xlabel('iteration');
    end
    subplot(6, 4, (fn-1)*4 + 3);
    plot(Tau_MCMC{fn});
    title('$\tau$', 'Interpreter', 'Latex');
    if fn < L_f
        set(gca, 'xticklabel', []);
    else
        xlabel('iteration');
    end
    subplot(6, 4, (fn-1)*4 + 4);
    plot(Rho_MCMC{fn});
    title('$\rho$', 'Interpreter', 'Latex');
    
    if fn < L_f
        set(gca, 'xticklabel', []);
    else
        xlabel('iteration');
    end
end
    
exportgraphics(gcf,[foldername '/MCMC_samples.pdf']);

%% Draw the 2D histograms
for fn = 1:L_f
 % Plot and save
    filename = filenames{fn};
    fc = fc + 1;
    f = figure(fc);
    set(gcf, 'Visible', 'on'); % Hide it
    f.Position = [100 100 200 170];
    set(gcf, 'renderer', 'painters');
    colormap(gray);
    histogram2(Eps_DP_MCMC{fn}(end/2:end), S_MCMC{fn}(end/2:end), 'DisplayStyle','tile','ShowEmptyBins','on');
    xlabel('$\epsilon$', 'Interpreter', 'Latex', 'Fontsize', 14);
    ylabel('$s$', 'Interpreter', 'Latex', 'Fontsize', 14);

    exportgraphics(gcf,sprintf('%s/histogram_%s.pdf',foldername, filename));
    close(f);
end