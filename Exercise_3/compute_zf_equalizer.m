function f_zf = compute_zf_equalizer(h_ls, K, delta)
    % Zero-Forcing Equalizer Computation
    % Inputs:
    %   h_ls: Channel impulse response (LS estimate)
    %   K: Equalizer length parameter
    %   delta: Delay index
    % Output:
    %   f_zf: ZF equalizer coefficients
    
    zf_length = 4*K + 1;
    g_length = length(h_ls) + zf_length - 1;
    
    % Unit vector with 1 at position delta
    e = zeros(g_length, 1);
    e(delta +1 ) = 1;
    
    % Construct Toeplitz matrix
    col_input = [h_ls; zeros(g_length - length(h_ls), 1)];
    row_input = [h_ls(1); zeros(g_length - length(h_ls), 1)];
    h_zf = toeplitz(col_input, row_input);
    
    % ZF equalizer solution
    f_zf = (h_zf' * h_zf) \ (h_zf' * e);
end