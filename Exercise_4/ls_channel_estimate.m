function h_ls = ls_channel_estimate(Yseq, train, M)
% Least squares estimation of the discrete equivalent channel of length M
%
%   y = A h + v   ->   h_LS = (A^H A)^{-1} A^H y
%
% Yseq  : symbol-spaced output sequence (after synchronization)
% train : training symbols, they are the first Ntr symbols of the packet

Ntr  = length(train);
Amat = training_matrix(train, M);
y    = Yseq(M:Ntr);                      % outputs y_{M-1},...,y_{Ntr-1}

h_ls = (Amat' * Amat) \ (Amat' * y);     % ' is the conjugate transpose (^H)
end
