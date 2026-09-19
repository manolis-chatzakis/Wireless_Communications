function [Ed, d_opt] = energy_Synchronization(Z, over, len, d_total)
% Energy statistic of the symbol-spaced subsequences, for MATLAB (1-based) delays
%
%   Ed(d) = sum_{n=0}^{len-1} |Z(d + n*over)|^2 ,   d = 1,...,d_total
%
% It is the same statistic as in ENERGY_SYNC, but the delays are expressed with
% MATLAB indices, so that d_opt can be used directly to sample the sequence Z.
%
% Z       : oversampled sequence (matched filter output, or the estimated
%           composite channel h_hat)
% over    : oversampling factor T/Ts
% len     : length of the symbol-spaced subsequence (N for the packet, M for
%           the discrete equivalent channel)
% d_total : number of candidate delays (search over d = 1,...,d_total)
%
% Ed      : the statistic, d_total x 1
% d_opt   : argmax(Ed), 1-based index of the best timing phase

Z  = Z(:);
Ed = energy_sync(Z, over, len, 0:d_total-1);   % d_values are 0-based -> +1 inside

[~, d_opt] = max(Ed);
end
