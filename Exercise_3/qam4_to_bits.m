function bits = qam4_to_bits(symbols)


    real_bits = (real(symbols) >= 0);  % <0 → 0, else 1
    imag_bits = (imag(symbols) >= 0);  % <0 → 0, else 1

    
    L = length(symbols);
    bits = zeros(2*L, 1);
    bits(1:2:end) = real_bits(:);
    bits(2:2:end) = imag_bits(:);

    
end
