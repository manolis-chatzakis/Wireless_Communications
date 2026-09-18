function [BER_ZF, BER_LS] = compute_ber_zf_ls(SNR_db, h_norm, n1, n2, K, delta, N, M_Pack, training_symbols, case_h )
    % Inputs: SNR_db (vector), h_norm (channel), n1/n2 (training indices), K (memory), delta (delay), N (data syms), M_Pack (Monte Carlo), training_symbols 
    % Outputs: BER_ZF, BER_LS (vectors) 

    BER_ZF = zeros(length(SNR_db), 1); 
    BER_LS = zeros(length(SNR_db), 1); 
    traing_bits =  qam4_to_bits(training_symbols);
    h = h_norm;
    for i_SNR = 1:length(SNR_db)
        SNR_lin = 10^(SNR_db(i_SNR)/10); 
        errors_ZF = 0; errors_LS = 0; 
        totalbits = 0; 

        

        for m = 1:M_Pack-1
            if(case_h ==1)
                L = K + 1;  % Vector length
                var = 1 / L;  % Per-entry variance (E[|h_i|^2] = var)
                h = sqrt(var / 2) * (randn(L, 1) + 1j * randn(L, 1));
            end

            % Packet generation 
            bits_all = randi([0 1], 2*N, 1); 
            s_package = bits_to_4qam(bits_all); 
            s_package(n1:n2) = training_symbols; 
            % Noisy output 
            Py = 2 * mean(abs(s_package).^2); 
            N0 = Py / SNR_lin; 
            y_clean = conv(s_package, h); 
            noise = sqrt(N0/2) * (randn(length(y_clean),1) + 1j*randn(length(y_clean),1));
            y_n = y_clean + noise; 
            
            % Channel estimation 
            y_train = y_n(n1+K : n2); 
            col_input = s_package(n1+K : n2); 
            row_input = s_package(n1+K : -1 : n1); 
            X_train = toeplitz(col_input, row_input); 
            h_ls = (X_train' * X_train) \ (X_train' * y_train); 
            
            %---------------------------------------------------
                s_packag_data = s_package([1:n1-1, n2+1:N]);
                bits_data = qam4_to_bits(s_packag_data); 

            %--------------------------------------------------
            % ZF 
            f_zf = compute_zf_equalizer(h_ls, K, delta); 
            r_zf = conv(y_n, f_zf); 
            s_h_zf = r_zf(delta+1 : delta + N); 

            s_h_zf_data = s_h_zf([1:n1-1, n2+1:N]);
            s_h_zf_train = s_h_zf(n1:n2);

            bits_hat_zf_data = qam4_to_bits(s_h_zf_data); 
            bits_hat_zf_train = qam4_to_bits(s_h_zf_train);

            train_error_zf = sum(bits_hat_zf_train ~= traing_bits);
            data_error_zf = sum(bits_hat_zf_data ~=bits_data );
            errors_ZF = errors_ZF  + data_error_zf+train_error_zf ;
         
            
            % LS 
            K_eq = length(f_zf)-1;
            f_ls = compute_ls_equalizer(y_n, n1, n2, K_eq, delta, training_symbols); 
            r_ls = conv(y_n, f_ls); 
            s_h_ls = r_ls(delta+1 : delta + N); 
    
            s_h_ls_data = s_h_ls([1:n1-1, n2+1:N]);
            s_h_ls_train = s_h_ls(n1:n2);

            bits_hat_ls_data = qam4_to_bits(s_h_ls_data); 
            bits_hat_ls_train = qam4_to_bits(s_h_ls_train); 

            train_error_ls = sum(bits_hat_ls_train ~= traing_bits);
            data_error_ls = sum(bits_hat_ls_data ~=bits_data );
            errors_LS = errors_LS  + data_error_ls+train_error_ls;

            
            totalbits = totalbits + length(bits_hat_ls_data) + length(s_h_ls_train); 
        end
        BER_ZF(i_SNR) = errors_ZF / totalbits; 
        BER_LS(i_SNR) = errors_LS / totalbits; 
    end
end 