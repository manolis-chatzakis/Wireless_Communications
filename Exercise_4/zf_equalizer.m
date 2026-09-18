function f = zf_equalizer(h, L, delta)
% Zero forcing (least squares) equalizer of length L, eq. (4.17)-(4.18)
%
%   H f = g ,   f_ZF = (H^H H)^{-1} H^H e_delta
%
% h     : discrete channel (length M)
% L     : equalizer length
% delta : desired delay (0-based), e_delta has 1 at position delta

M = length(h);

% Convolution matrix: column j is the channel shifted down by j-1 positions
H = zeros(M+L-1, L);
for j = 1:L
    H(j:j+M-1, j) = h;
end

e_delta = zeros(M+L-1, 1);
e_delta(delta+1) = 1;                    % +1 for MATLAB indexing

f = (H' * H) \ (H' * e_delta);
end
