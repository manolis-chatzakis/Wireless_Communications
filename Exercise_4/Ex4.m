%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Emmanouil Thomas Chatzakis                  %
%                2021030061                                  %
%                                                            %
%               Wireless Communications                      %
%     Exercise 4 - Synchronization, CFO and channel          %
%                  estimation                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;

rng(2021030061);        % fixed seed so that the results of the report are reproducible

% keep all the printed results in a text file
if exist('results_log.txt', 'file'); delete('results_log.txt'); end
diary('results_log.txt');

%% Parameters
T     = 1;              % symbol period
over  = 10;             % oversampling factor T/Ts
Ts    = T/over;         % sampling period
beta  = 0.4;            % roll-off factor of the SRRC
A     = 3;              % half length of the SRRC in symbol periods
N     = 200;            % symbols per packet
Ntr   = 50;             % training symbols (the first Ntr symbols of the packet)
sA2   = 2;              % sigma_A^2 = E[|A_n|^2] for 4-QAM

% SRRC pulse, same filter at the transmitter (g_T) and the receiver (g_R)
g = srrc_pulse(T, Ts, A, beta);
g = g(:);               % column vector, length 2*A*over+1 = 61

%% ======================================================================
%  PART 1 : ideal channel
%  ======================================================================

% 1.1 Packet of 4-QAM symbols A_0,...,A_{N-1}
bits  = randi([0 1], 2*N, 1);
An    = bits_to_4qam(bits);           % all the symbols of the packet
train = An(1:Ntr);                    % training symbols (known at the receiver)

% A(t) = sum_n A_n g_T(t - nT), computed at t = k*Ts
X = zeros(N*over, 1);
X(1:over:end) = An;                   % symbol A_n at t = nT (every over samples)
At = conv(X, g);                      % low-pass equivalent input of the channel

figure;
t_axis = (0:length(At)-1)*Ts;
plot(t_axis, real(At), 'b'); hold on;
plot(t_axis, imag(At), 'r');
xlim([0 30]); grid on;
xlabel('t / T'); ylabel('A(t)');
legend('Re\{A(t)\}', 'Im\{A(t)\}');
title('Low-pass equivalent channel input A(t) (first 30 T)');
save_figure('p1_At');

% Two cases: 1.2 c(t) = delta(t)   and   1.8 c(t) = c*delta(t) with complex c
c_rand   = (randn + 1j*randn)/sqrt(2);
c_values = [1, c_rand];
tags     = {'c1', 'crand'};
fprintf('===== PART 1 =====\n');
fprintf('Complex channel coefficient for 1.8: c = %.4f %+.4fj  (|c| = %.4f, angle = %.2f deg)\n', ...
        real(c_rand), imag(c_rand), abs(c_rand), angle(c_rand)*180/pi);

for ic = 1:2
    c   = c_values(ic);
    tag = tags{ic};
    fprintf('\n--- Part 1, case %s ---\n', tag);

    % 1.2 Noiseless ideal channel: output = c * A(t)
    Rt = c * At;

    % 1.3 Matched filter g_R(t) (time at its output starts at t = 0)
    Y = conv(Rt, g) * Ts;             % * Ts approximates the continuous convolution

    % 1.4 Composite analog channel h(t) = g_T(t) * c(t) * g_R(t)
    h = c * conv(g, g) * Ts;
    th = (0:length(h)-1)*Ts;
    figure;
    plot(th, abs(h), 'b', 'LineWidth', 1.5); grid on;
    xlabel('t / T'); ylabel('|h(t)|');
    title(sprintf('Composite analog channel |h(t)|, c = %.2f%+.2fj', real(c), imag(c)));
    save_figure(['p1_composite_' tag]);
    [~, i_peak] = max(abs(h));
    fprintf('Peak of |h(t)| at sample %d (t = %.1f T), |h| = %.4f\n', i_peak-1, (i_peak-1)*Ts, abs(h(i_peak)));

    % 1.5 Synchronization with the energy of the packet, d = 0,...,4*A*over-1
    d_values = 0:4*A*over-1;
    Ed = energy_sync(Y, over, N, d_values);
    [Ed_max, i_max] = max(Ed);
    d_energy = d_values(i_max);

    figure;
    plot(d_values, Ed, 'b.-'); hold on;
    plot(d_energy, Ed_max, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
    grid on; xlabel('d (samples)'); ylabel('E_d');
    title(sprintf('Energy statistic E_d, d^* = %d', d_energy));
    save_figure(['p1_Ed_' tag]);

    Ed_sorted = sort(Ed, 'descend');
    fprintf('Energy sync: d* = %d, E_max = %.3f, second largest E_d = %.3f\n', ...
            d_energy, Ed_sorted(1), Ed_sorted(2));

    % 1.6 Synchronization with the training symbols
    Cd = training_corr(Y, train, over, d_values);
    h_hat = Cd / (Ntr*sA2);           % estimate of the oversampled composite channel
    [~, i_max] = max(abs(Cd));
    d_train = d_values(i_max);

    figure;
    plot(d_values, abs(h_hat), 'b', 'LineWidth', 1.5); hold on;
    plot(d_values, abs(h(d_values+1)), 'r--', 'LineWidth', 1.5);
    grid on; xlabel('d (samples)'); ylabel('magnitude');
    legend('|C_d| / (N_{tr}\sigma_A^2)', '|h(dT_s)|');
    title(sprintf('Training statistic vs composite channel, d^* = %d', d_train));
    save_figure(['p1_Cd_' tag]);
    fprintf('Training sync: d* = %d\n', d_train);

    % 1.7 Symbol-spaced output sequence of length N (using training sync)
    r = Y(d_train + (0:N-1)*over + 1);
    plot_constellation(r, sprintf('Symbol-spaced output, c = %.2f%+.2fj', real(c), imag(c)));
    save_figure(['p1_output_' tag]);

    fprintf('max |r_k - c*A_k| = %.2e\n', max(abs(r - c*An)));
end

%% ======================================================================
%  PART 2 : ideal channel with CFO
%  ======================================================================
fprintf('\n===== PART 2 =====\n');

DFs = 1e-3;                 % DF*Ts
Df  = DFs*over;             % DF*T (CFO per symbol)
fprintf('DFs = DF*Ts = %.4g,  Df = DF*T = %.4g\n', DFs, Df);

% case a: c = 1, phi = 0          (2.1 - 2.3)
% case b: complex c, random phase (2.4)
phi_rand   = 2*pi*rand;
c2_values  = [1, c_rand];
phi_values = [0, phi_rand];
tags2      = {'a', 'b'};
fprintf('Case b: c = %.4f %+.4fj, phi = %.4f rad\n', real(c_rand), imag(c_rand), phi_rand);

% theoretical attenuation of |C_d| due to the CFO, |alpha_d| / (Ntr*sigma_A^2), eq. (5.28)
alpha_ratio = abs(sum(exp(1j*2*pi*Df*(0:Ntr-1)))) / Ntr;
fprintf('Theoretical |alpha|/(Ntr*sA2) = %.4f\n', alpha_ratio);

for ic = 1:2
    c   = c2_values(ic);
    phi = phi_values(ic);
    tag = tags2{ic};
    fprintf('\n--- Part 2, case %s ---\n', tag);

    % 2.1 - 2.2 Channel output with CFO: Y_n = c * exp(j(2*pi*DFs*n + phi)) * A(nTs)
    n  = (0:length(At)-1).';
    Rt = c * exp(1j*(2*pi*DFs*n + phi)) .* At;

    % 2.3 Matched filter, composite channel (the same as in part 1)
    Y = conv(Rt, g) * Ts;
    h = c * conv(g, g) * Ts;

    % Energy synchronization
    d_values = 0:4*A*over-1;
    Ed = energy_sync(Y, over, N, d_values);
    [Ed_max, i_max] = max(Ed);
    d_energy = d_values(i_max);

    figure;
    plot(d_values, Ed, 'b.-'); hold on;
    plot(d_energy, Ed_max, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
    grid on; xlabel('d (samples)'); ylabel('E_d');
    title(sprintf('Energy statistic E_d with CFO, d^* = %d', d_energy));
    save_figure(['p2_Ed_' tag]);
    fprintf('Energy sync: d* = %d\n', d_energy);

    % Training synchronization
    Cd = training_corr(Y, train, over, d_values);
    h_hat = Cd / (Ntr*sA2);
    [~, i_max] = max(abs(Cd));
    d_train = d_values(i_max);

    figure;
    plot(d_values, abs(h_hat), 'b', 'LineWidth', 1.5); hold on;
    plot(d_values, abs(h(d_values+1)), 'r--', 'LineWidth', 1.5);
    grid on; xlabel('d (samples)'); ylabel('magnitude');
    legend('|C_d| / (N_{tr}\sigma_A^2)', '|h(dT_s)|');
    title(sprintf('Training statistic with CFO, d^* = %d', d_train));
    save_figure(['p2_Cd_' tag]);
    fprintf('Training sync: d* = %d, max|h_hat|/max|h| = %.4f\n', d_train, max(abs(h_hat))/max(abs(h)));

    % Symbol-spaced output sequence of length N
    % We use the energy synchronization: the CFO multiplies every sample with a
    % complex exponential of magnitude 1, so the energy statistic does not change.
    % The peak of |C_d| is attenuated by |alpha| and it is very flat around the maximum.
    d_sync = d_energy;
    r = Y(d_sync + (0:N-1)*over + 1);
    plot_constellation(r, 'Symbol-spaced output with CFO (no correction)');
    save_figure(['p2_output_' tag]);

    % 2.5 CFO estimation with the training symbols (Chapter 6)
    Nfft = 2^16;
    [Df_hat, f_axis, Zmag] = cfo_estimate_fft(r(1:Ntr), train, Nfft);

    figure;
    plot(f_axis, Zmag, 'b', 'LineWidth', 1.2); hold on;
    plot(Df_hat, max(Zmag), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
    xlim([-0.1 0.1]); grid on;
    xlabel('digital frequency f (cycles/symbol)'); ylabel('|F\{z_l\}|');
    title(sprintf('Fourier transform of z_l, peak at f = %.5f', Df_hat));
    save_figure(['p2_cfo_fft_' tag]);
    fprintf('CFO estimate: Df_hat = %.6f (true %.6f), DFs_hat = %.3e (true %.3e)\n', ...
            Df_hat, Df, Df_hat/over, DFs);

    % 2.6 CFO correction: r~_k = r_k * exp(-j*2*pi*Df_hat*k)
    k = (0:N-1).';
    r_corr = r .* exp(-1j*2*pi*Df_hat*k);
    plot_constellation(r_corr, 'Symbol-spaced output after CFO correction');
    save_figure(['p2_cfo_corrected_' tag]);

    % 2.7 LS estimation of the discrete channel (one coefficient)
    %     h = (a^H a)^{-1} a^H r~   (' is the conjugate transpose)
    h0_ls = (train' * r_corr(1:Ntr)) / (train' * train);
    fprintf('LS channel: h0 = %.4f %+.4fj  (|h0| = %.4f, angle = %.2f deg)\n', ...
            real(h0_ls), imag(h0_ls), abs(h0_ls), angle(h0_ls)*180/pi);

    % 2.8 Remove the channel
    r_eq = r_corr / h0_ls;
    plot_constellation(r_eq, 'Output after CFO and channel correction');
    save_figure(['p2_equalized_' tag]);

    A_hat = sign(real(r_eq)) + 1j*sign(imag(r_eq));
    fprintf('Symbol errors: %d / %d, max |r_eq - A| = %.3f\n', sum(A_hat ~= An), N, max(abs(r_eq - An)));

    % Check: the same steps if we start from the training synchronization point
    if d_train ~= d_energy
        r_t  = Y(d_train + (0:N-1)*over + 1);
        Df_t = cfo_estimate_fft(r_t(1:Ntr), train, Nfft);
        r_t  = r_t .* exp(-1j*2*pi*Df_t*k);
        r_t  = r_t / ((train' * r_t(1:Ntr)) / (train' * train));
        fprintf('With d_train = %d instead: Df_hat = %.6f, max |r_eq - A| = %.3f\n', ...
                d_train, Df_t, max(abs(r_t - An)));
    end
end

%% ======================================================================
%  PART 3 : non-ideal channel without CFO
%  ======================================================================
fprintf('\n===== PART 3 =====\n');

% 3.1 Same packet as in part 1 (An, At)

% 3.2 Physical channel c(t) = c0 delta(t) + c1 delta(t - K Ts)
K  = randi([1, 4*over]);
c0 = (randn + 1j*randn)/sqrt(2);
c1 = (randn + 1j*randn)/sqrt(2);
c_phys = zeros(K+1, 1);
c_phys(1)   = c0;
c_phys(K+1) = c1;
fprintf('K = %d samples (%.1f T), c0 = %.4f %+.4fj, c1 = %.4f %+.4fj\n', ...
        K, K*Ts, real(c0), imag(c0), real(c1), imag(c1));

M     = 10;             % assumed length of the discrete equivalent channel
L     = 5*M;            % ZF equalizer length
delta = M;              % ZF delay

% 3.3 Channel output, matched filter and composite channel
Rt3 = conv(At, c_phys);               % output of the physical channel (noiseless)
Y   = conv(Rt3, g) * Ts;
h   = conv(conv(g, c_phys), g) * Ts;  % h(t) = g_T * c * g_R
th  = (0:length(h)-1)*Ts;

figure;
plot(th, abs(h), 'b', 'LineWidth', 1.5); grid on;
xlabel('t / T'); ylabel('|h(t)|');
title(sprintf('Composite analog channel |h(t)|, K = %d', K));
save_figure('p3_composite');

% 3.4 Training correlation for d = 0,...,(4*A*over+1)+(K+2)-2
d_max    = (4*A*over + 1) + (K + 2) - 2;
d_values = 0:d_max;
Y_pad    = [Y; zeros(N*over, 1)];     % zeros after the end of the received signal
h_pad    = [h; zeros(N*over, 1)];

Cd = training_corr(Y_pad, train, over, d_values);
h_hat = Cd / (Ntr*sA2);               % estimate of h(dTs), eq. (5.20)

figure;
plot(d_values, abs(h_hat), 'b', 'LineWidth', 1.5); hold on;
plot(d_values, abs(h_pad(d_values+1)), 'r--', 'LineWidth', 1.5);
grid on; xlabel('d (samples)'); ylabel('magnitude');
legend('|C_d| / (N_{tr}\sigma_A^2)', '|h(dT_s)|');
title('Training statistic vs composite channel (non-ideal channel)');
save_figure('p3_Cd');

% 3.5 Energy of length-M symbol-spaced subsequences of h_hat, eq. (5.21)
%     we need d + (M-1)*over <= d_max so that all samples of h_hat exist
dE_values = 0:(d_max - (M-1)*over);
Ed      = energy_sync(h_hat, over, M, dE_values);
Ed_true = energy_sync(h, over, M, dE_values);     % same quantity with the true h (for comparison)
[Ed_max, i_max] = max(Ed);
d_star = dE_values(i_max);
[~, i_true] = max(Ed_true);
fprintf('d range for E_d: 0,...,%d. d* (estimated h) = %d, d* (true h) = %d\n', ...
        dE_values(end), d_star, dE_values(i_true));
same_phase = mod(dE_values, over) == mod(d_star, over);
fprintf('E_d (true h) at d = %s : %s\n', mat2str(dE_values(same_phase)), mat2str(round(Ed_true(same_phase).', 4)));

figure;
plot(dE_values, Ed, 'b.-'); hold on;
plot(dE_values, Ed_true, 'r--');
plot(d_star, Ed_max, 'ko', 'MarkerSize', 8, 'LineWidth', 1.5);
grid on; xlabel('d (samples)'); ylabel('E_d');
legend('E_d from estimated h', 'E_d from true h', 'd^*', 'Location', 'best');
title(sprintf('Channel energy E_d for M = %d, d^* = %d', M, d_star));
save_figure('p3_Ed');

% symbol-spaced output of length N+M-1, eq. (5.24)
Yseq = Y_pad(d_star + (0:N+M-2)*over + 1);

% the true discrete equivalent channel h_m = h(d*Ts + mT), eq. (5.11)
h_disc = h_pad(d_star + (0:M-1)*over + 1);

figure;
plot(th, abs(h), 'b', 'LineWidth', 1.2); hold on;
stem((d_star + (0:M-1)*over)*Ts, abs(h_disc), 'r', 'filled');
grid on; xlabel('t / T'); ylabel('magnitude');
legend('|h(t)|', '|h_m|, discrete equivalent');
title('Composite analog channel and discrete equivalent channel');
save_figure('p3_discrete_channel');

plot_constellation(Yseq, 'Symbol-spaced output of the non-ideal channel');
save_figure('p3_output');

% 3.5 (second) LS estimation of the discrete equivalent channel of length M
h_ls = ls_channel_estimate(Yseq, train, M);
fprintf('LS channel error ||h_ls - h_disc|| / ||h_disc|| = %.3e\n', norm(h_ls - h_disc)/norm(h_disc));

figure;
subplot(2,1,1);
stem(0:M-1, real(h_disc), 'bo', 'LineWidth', 1.2); hold on;
stem(0:M-1, real(h_ls), 'rx--', 'LineWidth', 1.2);
grid on; xlabel('m'); ylabel('Re\{h_m\}');
legend('true', 'LS estimate'); title('Real part');
subplot(2,1,2);
stem(0:M-1, imag(h_disc), 'bo', 'LineWidth', 1.2); hold on;
stem(0:M-1, imag(h_ls), 'rx--', 'LineWidth', 1.2);
grid on; xlabel('m'); ylabel('Im\{h_m\}');
legend('true', 'LS estimate'); title('Imaginary part');
save_figure('p3_ls_channel');

% 3.6 ZF equalizer of length L = 5M with delay delta = M
f_zf = zf_equalizer(h_ls, L, delta);
g_total = conv(h_ls, f_zf);           % overall response channel - equalizer

figure;
stem(0:length(g_total)-1, abs(g_total), 'b', 'LineWidth', 1.2); grid on;
xlabel('n'); ylabel('|(h * f)_n|');
title(sprintf('Overall response channel-ZF equalizer, \\delta = %d', delta));
save_figure('p3_zf_total');
fprintf('ZF: |g_delta| = %.4f, max |g_n| (n ~= delta) = %.4f\n', ...
        abs(g_total(delta+1)), max(abs(g_total([1:delta, delta+2:end]))));

% 3.7 Equalize, drop the first delta samples and keep the next N
z = conv(Yseq, f_zf);
z_out = z(delta+1 : delta+N);
plot_constellation(z_out, 'Output of the ZF equalizer');
save_figure('p3_equalized');

A_hat = sign(real(z_out)) + 1j*sign(imag(z_out));
fprintf('Symbol errors: %d / %d, MSE = %.3e\n', sum(A_hat ~= An), N, mean(abs(z_out - An).^2));

%% ======================================================================
%  PART 4 : non-ideal channel with CFO
%  ======================================================================
fprintf('\n===== PART 4 =====\n');

% 4.1 Same packet and same physical channel as in part 3, now with CFO
DFs4 = 1e-3;
Df4  = DFs4*over;
phi4 = 2*pi*rand;
fprintf('DFs = %.4g (Df = %.4g), phi = %.4f rad\n', DFs4, Df4, phi4);

n   = (0:length(Rt3)-1).';
Rt4 = exp(1j*(2*pi*DFs4*n + phi4)) .* Rt3;   % received signal with CFO
Y   = conv(Rt4, g) * Ts;
Y_pad = [Y; zeros(N*over, 1)];

% Training correlation (same d range as in part 3)
Cd = training_corr(Y_pad, train, over, d_values);
h_hat = Cd / (Ntr*sA2);

figure;
plot(d_values, abs(h_hat), 'b', 'LineWidth', 1.5); hold on;
plot(d_values, abs(h_pad(d_values+1)), 'r--', 'LineWidth', 1.5);
grid on; xlabel('d (samples)'); ylabel('magnitude');
legend('|C_d| / (N_{tr}\sigma_A^2)', '|h(dT_s)|');
title('Training statistic vs composite channel (non-ideal channel, CFO)');
save_figure('p4_Cd');

% Channel energy E_d and synchronization (the unknown |alpha| is common to all d)
Ed = energy_sync(h_hat, over, M, dE_values);
[Ed_max, i_max] = max(Ed);
d_star4 = dE_values(i_max);
fprintf('d* with CFO = %d (d* without CFO = %d)\n', d_star4, d_star);

figure;
plot(dE_values, Ed, 'b.-'); hold on;
plot(dE_values, Ed_true, 'r--');
plot(d_star4, Ed_max, 'ko', 'MarkerSize', 8, 'LineWidth', 1.5);
grid on; xlabel('d (samples)'); ylabel('E_d');
legend('E_d from estimated h (CFO)', 'E_d from true h', 'd^*', 'Location', 'best');
title(sprintf('Channel energy E_d with CFO, d^* = %d', d_star4));
save_figure('p4_Ed');

Yseq4 = Y_pad(d_star4 + (0:N+M-2)*over + 1);
h_disc4 = h_pad(d_star4 + (0:M-1)*over + 1);    % discrete channel without CFO (for comparison)

plot_constellation(Yseq4, 'Symbol-spaced output (non-ideal channel, CFO)');
save_figure('p4_output');

% What happens if we ignore the CFO and apply the steps of part 3 directly
h_ls_noCFO = ls_channel_estimate(Yseq4, train, M);
f_noCFO    = zf_equalizer(h_ls_noCFO, L, delta);
z_noCFO    = conv(Yseq4, f_noCFO);
z_noCFO    = z_noCFO(delta+1 : delta+N);
plot_constellation(z_noCFO, 'ZF output without CFO correction');
save_figure('p4_equalized_noCFO');
A_hat = sign(real(z_noCFO)) + 1j*sign(imag(z_noCFO));
fprintf('Without CFO correction: symbol errors %d / %d\n', sum(A_hat ~= An), N);

% Joint LS estimation of CFO and channel (Section 5.3.1)
df_grid = -0.5 : 1e-5 : 0.5-1e-5;
[Df_hat4, h_star, cost] = joint_cfo_channel_ls(Yseq4, train, M, df_grid);
fprintf('Joint LS: Df_hat = %.5f (true %.5f), DFs_hat = %.3e\n', Df_hat4, Df4, Df_hat4/over);

figure;
subplot(2,1,1);
plot(df_grid, cost, 'b'); grid on;
xlabel('\Deltaf (cycles/symbol)'); ylabel('g(\Deltaf)');
title('Cost g(\Deltaf) = y^H\Gamma P \Gamma^H y');
subplot(2,1,2);
plot(df_grid, cost, 'b', 'LineWidth', 1.2); hold on;
plot(Df_hat4, max(cost), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
xlim([-0.05 0.05]); grid on;
xlabel('\Deltaf (cycles/symbol)'); ylabel('g(\Deltaf)');
title(sprintf('Zoom, \\Deltaf^* = %.5f', Df_hat4));
save_figure('p4_cfo_cost');

% CFO correction, eq. (5.42)
k = (0:N+M-2).';
Yseq4_corr = Yseq4 .* exp(-1j*2*pi*Df_hat4*k);
plot_constellation(Yseq4_corr, 'Symbol-spaced output after CFO correction');
save_figure('p4_cfo_corrected');

% LS estimation of the discrete channel on the corrected sequence (step 3.5)
h_ls4 = ls_channel_estimate(Yseq4_corr, train, M);
fprintf('||h_ls4 - h*|| = %.3e (LS after correction vs joint LS)\n', norm(h_ls4 - h_star));

figure;
stem(0:M-1, abs(h_disc4), 'bo', 'LineWidth', 1.2); hold on;
stem(0:M-1, abs(h_ls4), 'rx--', 'LineWidth', 1.2);
grid on; xlabel('m'); ylabel('|h_m|');
legend('true (without CFO)', 'LS estimate (CFO corrected)');
title('Magnitude of the discrete equivalent channel');
save_figure('p4_ls_channel');

% Phase difference between estimate and true channel (common for all taps)
phase_diff = angle(h_ls4 ./ h_disc4) * 180/pi;
fprintf('Phase of h_ls4 ./ h_disc4 (deg): %s\n', mat2str(round(phase_diff.', 1)));

% ZF equalizer (L = 5M, delta = M)
f_zf4 = zf_equalizer(h_ls4, L, delta);
g_total4 = conv(h_ls4, f_zf4);

figure;
stem(0:length(g_total4)-1, abs(g_total4), 'b', 'LineWidth', 1.2); grid on;
xlabel('n'); ylabel('|(h * f)_n|');
title(sprintf('Overall response channel-ZF equalizer (CFO), \\delta = %d', delta));
save_figure('p4_zf_total');

z4 = conv(Yseq4_corr, f_zf4);
z4_out = z4(delta+1 : delta+N);
plot_constellation(z4_out, 'ZF output after CFO correction');
save_figure('p4_equalized');

A_hat = sign(real(z4_out)) + 1j*sign(imag(z4_out));
fprintf('Symbol errors: %d / %d, MSE = %.3e\n', sum(A_hat ~= An), N, mean(abs(z4_out - An).^2));

% Extra check: the same steps if the packet starts from d* of part 3
fprintf('d* mod over: part 3 -> %d, part 4 -> %d\n', mod(d_star, over), mod(d_star4, over));
Ys   = Y_pad(d_star + (0:N+M-2)*over + 1);
Df_s = joint_cfo_channel_ls(Ys, train, M, df_grid);
Ys   = Ys .* exp(-1j*2*pi*Df_s*k);
f_s  = zf_equalizer(ls_channel_estimate(Ys, train, M), L, delta);
z_s  = conv(Ys, f_s);
z_s  = z_s(delta+1 : delta+N);
fprintf('Starting from d = %d instead: Df_hat = %.5f, MSE = %.3e\n', d_star, Df_s, mean(abs(z_s - An).^2));

diary off;
