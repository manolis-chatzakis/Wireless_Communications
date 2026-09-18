function C = training_corr(Y, train, over, d_values)
% Correlation of the oversampled output with the training symbols, eq. (5.18)
%
%   C_d = sum_{n=0}^{Ntr-1} conj(A_n) * Y_{d + n*over} ,  d in d_values
%
% Since E[C_d] = Ntr * sigma_A^2 * h(d*Ts)  (eq. 5.19), C_d/(Ntr*sigma_A^2)
% is an estimate of the oversampled composite channel (eq. 5.20).

Ntr = length(train);
C = zeros(length(d_values), 1);

for i = 1:length(d_values)
    d = d_values(i);
    idx = d + (0:Ntr-1)*over + 1;      % sampling instants dTs + nT (+1 for MATLAB)
    C(i) = sum(conj(train) .* Y(idx));
end
end
