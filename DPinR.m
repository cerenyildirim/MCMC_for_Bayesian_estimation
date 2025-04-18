function [Y] = DPinR(A, B, eps_DP, delta_DP)

% [Y] = DPinR(A, B, eps_DP, delta_DP)
% 
% This function determines if a point in [0, 1] x [0, 1] is in the region
% R(epsilon, delta) of admissible (alpha, beta) error probabilities.

Cond1 = A + exp(eps_DP)*B >= 1 - delta_DP;
Cond2 = B + exp(eps_DP)*A >= 1 - delta_DP;
Cond3 = A + exp(eps_DP)*B <= exp(eps_DP) + delta_DP;
Cond4 = A + exp(eps_DP)*B <= exp(eps_DP) + delta_DP;

Y = Cond1 & Cond2 & Cond3 & Cond4;