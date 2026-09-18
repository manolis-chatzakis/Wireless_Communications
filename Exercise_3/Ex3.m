clear all; close all; clc;

%% Part A
K = 4;
N = 200;

%A1 - Channel creation
h = (randn(K+1,1) + 1j*randn(K+1,1))/sqrt(2);
energy = sum(abs(h).^2);
h_norm = h / sqrt(energy);  % Normalization
%norm_energy = sum(abs(h_norm).^2);  % ~1

%A2
bits_all = randi([0 1], 2*N, 1);
symbol_package = bits_to_4qam(bits_all); 

%A3
n1 = 40;
n2 = 60;
training_symbols = symbol_package(n1:1:n2);  
L_train = length(training_symbols); % n2-n1+1 = 21

%A4
y = conv(symbol_package, h_norm);
%L_y = length(y);  % N+K

%A5       
Py = 2 * sum(abs(h_norm).^2); 

%A6 
SNR_db = 30;
SNR_linear = 10^(SNR_db/10);
N0 = Py / SNR_linear;
noise = sqrt(N0/2) * (randn(length(y),1) + 1j*randn(length(y),1));
y_n = y + noise;

%A7 
figure;
scatter(real(y), imag(y), 20, 'b', 'filled'); hold on;
scatter(real(y_n), imag(y_n), 20, 'r','filled');
grid on; axis equal;
xlabel('Re{y}'); ylabel('Im{y}');
legend('Without noise', 'With noise');
title(sprintf('Channel output, SNR = %d dB', SNR_db));
hold off;

%% Part B 
%B1-B2
y_train = y_n(n1+K : n2);  


col_input = symbol_package(n1+K : 1 : n2);  
row_input = symbol_package(n1+K : -1 : n1);  
X_train = toeplitz(col_input, row_input);  

%B3 - LS estimation
h_ls = (X_train' * X_train) \ (X_train' * y_train);
h_est_energy = sum(abs(h_ls).^2);

%B4
figure;
plot_dim = 0:K;
subplot(2,1,1);
stem(plot_dim, real(h_norm), 'bo-', 'LineWidth', 1.2); hold on;
stem(plot_dim, real(h_ls), 'rx--', 'LineWidth', 1.2);
grid on; xlabel('Tap k'); ylabel('Re{h_k}');
legend('True', 'LS estimate'); title('Real part');
hold off;
subplot(2,1,2);
stem(plot_dim, imag(h_norm), 'bo-', 'LineWidth', 1.2); hold on;
stem(plot_dim, imag(h_ls), 'ro--', 'LineWidth', 1.2);
grid on; xlabel('Tap k'); ylabel('Im{h_k}');
legend('True', 'LS estimate'); title('Imaginary part');
hold off;

%% PART C 
%C1
delta = K;
f_zf = compute_zf_equalizer(h_ls, K, delta);

%C2
g = conv(h_norm,f_zf);

figure;

subplot(2,1,1);
stem(0:length(g)-1, real(g), 'bo-','LineWidth',1.2);
grid on;
xlabel('n');
ylabel('Re\{g^{ZF}\}');
title('Real part of \{h * f_{ZF}\}');

subplot(2,1,2);
stem(0:length(g)-1, imag(g), 'ro-','LineWidth',1.2);
grid on;
xlabel('n');
ylabel('Im\{g^{ZF}\}');
title('Imaginary part of \{h * f_{ZF}\}');


%C3
r_zf = conv(y_n,f_zf);


%C4
start_idx = delta+1 ;
end_ind = delta + N ;
s_n = r_zf(start_idx : end_ind);

%c5
constellation = [ 1+1i; 1-1i; -1+1i; -1-1i ];
figure;
scatter(real(s_n), imag(s_n), 20, 'b', 'filled'); hold on;
scatter(real(constellation), imag(constellation), 20, 'r', 'filled'); hold on;


plot([0 0], ylim, 'k--', 'LineWidth', 1); hold on;
plot(xlim, [0 0], 'k--', 'LineWidth', 1); hold on;

grid on; axis equal;
xlabel('Re{y}'); ylabel('Im{y}');
title('Properly truncate the equalizer output');
hold off;


% C6 - LS Equalizer
K_eq = length(f_zf)- 1; 
f_ls = compute_ls_equalizer(y_n, n1, n2, K_eq, delta,training_symbols);

% Effective response
gLS = conv(h_norm, f_ls);
figure;
subplot(2,1,1);
stem(0:length(gLS)-1, real(gLS), 'bo-', 'LineWidth', 1.2); grid on;
xlabel('n'); ylabel('Re(g_{LS})'); title('Real part of h * f_{LS}');
subplot(2,1,2);
stem(0:length(gLS)-1, imag(gLS), 'ro-', 'LineWidth', 1.2); grid on;
xlabel('n'); ylabel('Im(g_{LS})'); title('Imaginary part of h * f_{LS}');

% Equalized symbols 
rLS = conv(y_n, f_ls); 
startidx_LS = delta + 1;
endind_LS = delta + N;
sn_LS = rLS(startidx_LS : endind_LS);
figure;
scatter(real(sn_LS), imag(sn_LS), 20, 'b', 'filled');
hold on;
scatter(real(constellation), imag(constellation), 50, 'r', 'filled');
plot([0 0], ylim, 'k--', 'LineWidth', 1);
plot(xlim, [0 0], 'k--', 'LineWidth', 1);
grid on; axis equal;
xlabel('Re(y)'); ylabel('Im(y)'); title('LS Equalizer Constellation');
hold off;


%C7

for delta = 0 : 1 : 4*K
    % Zero forcing eq
    f_zf = compute_zf_equalizer(h_ls, K, delta);
    r_zf = conv(y_n,f_zf);
    start_idx = delta+1 ;
    end_ind = delta + N-1 ;
    s_n_zf = r_zf(start_idx : end_ind);
    
    zf_error(delta+1) = mean(abs(s_n_zf(n1:n2) - training_symbols(1:1:end)).^2);

    % MMSE eq
    K_eq = length(f_zf) - 2; 
    f_ls = compute_ls_equalizer(y_n, n1, n2, K_eq, delta,training_symbols);
    rLS = conv(y_n, f_ls); 
    startidx_LS = delta + 1;
    endind_LS = delta + N  ;
    sn_LS = rLS(startidx_LS : endind_LS);
    
    MMSE_error(delta+1) =  mean(abs(sn_LS(n1:n2) - training_symbols(1:1:end)).^2);

end

[min_err_zf, idx_zf] = min(zf_error);
opt_d_zf = idx_zf - 1; % Because loop started at delta=0 but index starts at 1

[min_err_ls, idx_ls] = min(MMSE_error);
opt_d_ls = idx_ls - 1;


f_zf_opt = compute_zf_equalizer(h_ls, K, opt_d_zf);
f_ls_opt = compute_ls_equalizer(y_n, n1, n2, length(f_zf_opt)-1, opt_d_ls, training_symbols);

g_zf_opt = conv(h_norm, f_zf_opt);
figure;
subplot(2,1,1);
stem(0:length(g_zf_opt)-1, real(g_zf_opt), 'bo-', 'LineWidth', 1.2);
grid on;
xlabel('n');
ylabel('Re\{g^{ZF}_{opt}\}');
title(['Real part of \{h * f^{ZF}_{opt}\} (Delay \Delta = ' num2str(opt_d_zf) ')']);

subplot(2,1,2);
stem(0:length(g_zf_opt)-1, imag(g_zf_opt), 'ro-', 'LineWidth', 1.2);
grid on;
xlabel('n');
ylabel('Im\{g^{ZF}_{opt}\}');
title(['Imaginary part of \{h * f^{ZF}_{opt}\}']);

r_zf_opt = conv(y_n, f_zf_opt);
start_idx_zf = opt_d_zf + 1;
end_ind_zf = opt_d_zf + N;
s_n_zf_opt = r_zf_opt(start_idx_zf : end_ind_zf);

figure;
scatter(real(s_n_zf_opt), imag(s_n_zf_opt), 20, 'b', 'filled'); hold on;
scatter(real(constellation), imag(constellation), 20, 'r', 'filled'); hold on;
plot([0 0], ylim, 'k--', 'LineWidth', 1);
plot(xlim, [0 0], 'k--', 'LineWidth', 1);
grid on; axis equal;
xlabel('Re{y}'); ylabel('Im{y}');
title(['Optimal ZF Equalizer Output (Delay \Delta = ' num2str(opt_d_zf) ')']);
hold off;

g_ls_opt = conv(h_norm, f_ls_opt);
figure;
subplot(2,1,1);
stem(0:length(g_ls_opt)-1, real(g_ls_opt), 'bo-', 'LineWidth', 1.2); 
grid on;
xlabel('n'); ylabel('Re(g_{LS,opt})'); 
title(['Real part of h * f_{LS,opt} (Delay \Delta = ' num2str(opt_d_ls) ')']);

subplot(2,1,2);
stem(0:length(g_ls_opt)-1, imag(g_ls_opt), 'ro-', 'LineWidth', 1.2); 
grid on;
xlabel('n'); ylabel('Im(g_{LS,opt})'); 
title(['Imaginary part of h * f_{LS,opt}']);

r_ls_opt = conv(y_n, f_ls_opt); 
startidx_LS_opt = opt_d_ls + 1;
endind_LS_opt = opt_d_ls + N;
sn_LS_opt = r_ls_opt(startidx_LS_opt : endind_LS_opt);

figure;
scatter(real(sn_LS_opt), imag(sn_LS_opt), 20, 'b', 'filled');
hold on;
scatter(real(constellation), imag(constellation), 50, 'r', 'filled');
plot([0 0], ylim, 'k--', 'LineWidth', 1);
plot(xlim, [0 0], 'k--', 'LineWidth', 1);
grid on; axis equal;
xlabel('Re(y)'); ylabel('Im(y)'); 
title(['Optimal LS Equalizer Constellation (Delay \Delta = ' num2str(opt_d_ls) ')']);
hold off;

%% D
clear all; close all; clc;
% D1
K = 4;
N = 200;
h = (randn(K+1,1) + 1j*randn(K+1,1))/sqrt(2);
energy = sum(abs(h).^2);
h_norm = h / sqrt(energy);  % Normalization
%norm_energy = sum(abs(h_norm).^2);  % ~1

% creation of training symbols
n1 = 40;
n2 = 60;

training_bits = randi([0 1], 2*N, 1);
training_seq =  bits_to_4qam(training_bits);
training_symbols = training_seq(n1:n2); %training symbols

%D2
SNR_db = 2:2:30;
delta = K;
M_Pack = 1000;  %packets per SNR


[BER_ZF_D2, BER_LS_D2] = compute_ber_zf_ls(SNR_db, h_norm, n1, n2, K, delta, N, M_Pack, training_symbols, 0);

%D4

[BER_ZF_D4, BER_LS_D4] = compute_ber_zf_ls(SNR_db, h_norm, n1, n2, K, delta, N, M_Pack, training_symbols,1);

figure;
yscale log;
semilogy(SNR_db-2, BER_ZF_D2,      '-',  'LineWidth', 1.5); hold on;
semilogy(SNR_db-2, BER_LS_D2,      '-',  'LineWidth', 1.5);
semilogy(SNR_db-2, BER_ZF_D4, '--', 'LineWidth', 1.5);
semilogy(SNR_db-2, BER_LS_D4, '--', 'LineWidth', 1.5);
grid on;
xlabel('SNR in dB'); ylabel('Bit Error Rate');
legend('BER\_ZF','BER\_LS','BER\_ZF\_rand','BER\_LS\_rand','Location','northeast');
title('BER performance (ZF vs LS) - fixed and random channel');
hold off; 



