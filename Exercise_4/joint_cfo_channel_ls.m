function [Df_hat, h_hat, cost] = joint_cfo_channel_ls(Yseq, train, M, df_grid)
% Joint LS estimation of CFO and channel (Section 5.3.1)
%
%   y_k = exp(j*2*pi*Df*k) * sum_m h_m A_{k-m}
%   F(Df, h) = || y - Gamma(Df) A h ||^2                       (5.34)
%   Df*  = argmax  y^H Gamma(Df) P Gamma(Df)^H y               (5.40)
%   h*   = (A^H A)^{-1} A^H Gamma(Df*)^H y                     (5.41)
%
% The diagonal of Gamma uses the real time index k of each output
% (k = M-1,...,Ntr-1). This way the phase of h* agrees with the corrected
% sequence Y_k * exp(-j*2*pi*Df*k), k = 0,...,N+M-2  (5.42).

Ntr  = length(train);
Amat = training_matrix(train, M);
y    = Yseq(M:Ntr);                        % outputs that depend only on training
k    = (M-1:Ntr-1).';                      % their time indices (0-based)

P = Amat / (Amat' * Amat) * Amat';         % projection matrix (5.38)

cost = zeros(size(df_grid));
for i = 1:length(df_grid)
    v = exp(-1j*2*pi*df_grid(i)*k) .* y;   % Gamma(Df)^H y
    cost(i) = real(v' * P * v);            % g(Df)
end

[~, i_best] = max(cost);
Df_hat = df_grid(i_best);

y_corr = exp(-1j*2*pi*Df_hat*k) .* y;
h_hat  = (Amat' * Amat) \ (Amat' * y_corr);
end
