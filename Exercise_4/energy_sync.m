function E = energy_sync(Y, over, len, d_values)
% Energy of all symbol-spaced subsequences of length len
%
%   E_d = sum_{n=0}^{len-1} |Y_{d + n*over}|^2 ,  d in d_values
%
% Y        : oversampled signal, Y(1) corresponds to time index 0
% over     : oversampling factor T/Ts
% len      : length of the subsequence (N for the packet, M for the channel)
% d_values : candidate delays (in samples, starting from 0)
%
% The same function is used for
%   - the packet energy (Y = matched filter output, len = N)     -> Section 5.1
%   - the channel energy (Y = h_hat from training, len = M)      -> eq. (5.21)

E = zeros(length(d_values), 1);

for i = 1:length(d_values)
    d = d_values(i);
    idx = d + (0:len-1)*over + 1;      % +1 because MATLAB indices start from 1
    idx = idx(idx <= length(Y));       % after the end of the signal we only have zeros
    E(i) = sum(abs(Y(idx)).^2);
end
end
