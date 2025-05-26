function decision = LRT_normal(x, mu0, var0, mu1, var1, alpha)

% decision = LRT_normal(x, mu0, var0, mu1, var1, alpha)
% 
% Applies the decision rule of the LRT that compares two normal
% distributions with a single observation

R = (mu0/var0 - mu1/var1)/(1/var1 - 1/var0);
delta = mu0/sqrt(var0) + R/sqrt(var0);
if var0 > var1
    % decision = (x + R)^2/var0 <= ncx2inv(alpha, 1, delta^2);
    decision = ncx2cdf((x + R)^2/var0, 1, delta^2) <= alpha;
elseif var0 < var1
    % decision = (x + R)^2/var0 >= ncx2inv(1-alpha, 1, delta^2);
    decision = ncx2cdf((x + R)^2/var0, 1, delta^2) >= 1-alpha;
    
else
    decision = sign(mu0 - mu1)*(x - mu0)/sqrt(var0) > norminv(1-alpha);
end
