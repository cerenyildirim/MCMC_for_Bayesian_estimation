function [x, y] = gen_beta_DP(eps_DP, delta_DP)

c = (1-delta_DP)/(1 + exp(eps_DP));

x = [0, c, 1-delta_DP, 1];

cond1 = x < c;
cond2 = (x >= c) & (x < (1 - delta_DP));
cond3 = x > (1 - delta_DP);

y(cond1==1) = 1 - delta_DP - exp(eps_DP)*x(cond1);
y(cond2==1) = c - exp(-eps_DP)*(x(cond2)-c);
y(cond3==1) = 0;