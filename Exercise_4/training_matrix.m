function Amat = training_matrix(train, M)
% Matrix of the training symbols for LS channel estimation, eq. (4.9)
%
% Only the outputs y_k, k = M-1,...,Ntr-1 (0-based) depend ONLY on training symbols:
%   y_k = h_0 A_k + h_1 A_{k-1} + ... + h_{M-1} A_{k-M+1}
% so row i of the matrix is [A_k  A_{k-1} ... A_{k-M+1}].

Ntr = length(train);
Amat = zeros(Ntr-M+1, M);

for i = 1:Ntr-M+1
    k = i + M - 1;                       % MATLAB index of the output y_k
    Amat(i,:) = train(k:-1:k-M+1).';     % .' -> transpose WITHOUT conjugation
end
end
