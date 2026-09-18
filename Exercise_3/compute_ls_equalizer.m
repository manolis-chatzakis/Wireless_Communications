function f_ls = compute_ls_equalizer(y_n, n1, n2, K_eq, delta,training_symbols)
    % Least Squares Equalizer Computation
    % Inputs:
    %   y_n: Received signal
    %   n1, n2: Training sequence indices
    %   K_eq: LS equalizer length
    %   delta: Delay index
    % Output:
    %   f_ls: LS equalizer coefficients
    
    n1_eq = n1 + delta;
    n2_eq = n2 + delta;
    
    % Construct training data Toeplitz matrix
    col_y = y_n(n1_eq:1:n2_eq);
    row_y = y_n(n1_eq:-1:n1_eq-K_eq);
    Ytrain = toeplitz(col_y, row_y);
    
    
    % LS equalizer solution
    f_ls = (Ytrain' * Ytrain) \ (Ytrain' * training_symbols);
end