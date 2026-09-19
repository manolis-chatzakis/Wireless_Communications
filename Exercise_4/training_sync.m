function [C_d, corr_Y, d_opt_corr] = training_sync(Z_r, training_s, over, Ntr, d_total)
% Synchronization with the training symbols, eq. (5.18), for MATLAB (1-based) delays
%
%   C_d = sum_{n=0}^{Ntr-1} conj(A_n) * Z_r(d + n*over) ,   d = 1,...,d_total
%
% Since E[C_d] = Ntr * sigma_A^2 * h(d*Ts)  (5.19), the quantity
% C_d/(Ntr*sigma_A^2) is an estimate of the oversampled composite channel (5.20).
%
% Z_r        : matched filter output
% training_s : the Ntr training symbols (the first Ntr symbols of the packet)
% over       : oversampling factor T/Ts
% Ntr        : number of training symbols
% d_total    : number of candidate delays (search over d = 1,...,d_total)
%
% C_d        : the complex statistic, d_total x 1
% corr_Y     : |C_d|
% d_opt_corr : argmax|C_d|, 1-based index of the best timing phase

Z_r = Z_r(:);

% the last candidate delay needs the sample d_total + (Ntr-1)*over; after the
% end of the matched filter output the signal is zero
need_len = d_total + (Ntr-1)*over;
if length(Z_r) < need_len
    Z_r(end+1:need_len) = 0;
end

C_d    = training_corr(Z_r, training_s(1:Ntr), over, 0:d_total-1);
corr_Y = abs(C_d);

[~, d_opt_corr] = max(corr_Y);
end
