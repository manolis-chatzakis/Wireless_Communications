clear;
close all;
clc;

rng(2021030061); % fixed seed, so that the figures of the report are reproducible

T = 1;
over = 10;

Ts = T / over;

beta = 0.4; %roll-off factor
B = 3;      %half symbol length
Ntr = 50;
N = 200;

s_en = 2; %symbol energy

dFs = 1e-3;      % DF*Ts, normalized CFO per sample
Df  = dFs*over;  % DF*T,  normalized CFO per symbol

script_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(script_dir, 'figures', 'part2');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

%% 2
%2.1 (= steps 1.1 and 1.2)

data_bits = randi([0 1], 2*(N-Ntr), 1); %300
training_bits = randi([0 1], 2*Ntr, 1); %100

g(:,1) = srrc_pulse(T,Ts,B,beta);

data_s = bits_to_4qam(data_bits);    %150
trainig_s = bits_to_4qam(training_bits); %50

packet = [trainig_s;data_s]; % 200 symbols, the first 50 are training

packet_up = upsample(packet,over); %for correct conv later

A = conv(packet_up,g); % low pass equivalent of channel input

%% channel / phase cases
% 2.1 - 2.3 : c = 1, phi = 0
% 2.4       : c in C (step 1.8) and/or random phase phi ~ U[0,2*pi)

c_rand = (randn + 1i*randn)/sqrt(2);
phi_rand = 2*pi*rand;

c_values   = {1, c_rand, 1, c_rand};
phi_values = {0, 0, phi_rand, phi_rand};
tags = {'_ideal', '_complex', '_ideal_randphase', '_complex_randphase'};

fprintf('dFs = %.4g, Df = dFs*over = %.4g\n', dFs, Df);

for ic = 1:numel(c_values)

    c = c_values{ic};
    phi = phi_values{ic};
    tag = tags{ic};

    fprintf('\n--- case %s: c = %.4f %+.4fi, phi = %.4f rad ---\n', ...
            tag, real(c), imag(c), phi);

    Y_r = conv(A,c); %received from channel (noiseless)

    %2.2 CFO: multiply by exp(j*(2*pi*dFs*n + phi)), n = t/Ts

    n_ax = (0:length(Y_r)-1).';
    Y_cfo = Y_r .* exp(1i*(2*pi*dFs*n_ax + phi));

    %2.3 (= steps 1.3 - 1.7)

    Z_r = conv(Y_cfo,g)*Ts; %filtered

    % the CFO has unit modulus, so |composite channel| is the same as in part 1
    analog_channel = conv(conv(c,g),g)*Ts;
    x = length(analog_channel) - 1;
    figure;
    plot(0:Ts:x*Ts,abs(analog_channel));
    xlabel("Time");
    ylabel("Amplitude")
    grid on;
    title("Composite Analog Channel");
    saveas(gcf, fullfile(fig_dir, ['part2_2_3_analog_channel' tag '.png']));

    d_total = 4*B*over;

    [Ed,d_opt] = energy_Synchronization(Z_r, over, N, d_total);

    figure;
    plot(1:length(Ed), Ed); grid on;
    title('E_d (with CFO)');
    xlabel('d (samples)'); ylabel('E_d');
    saveas(gcf, fullfile(fig_dir, ['part2_2_3_energy_stat' tag '.png']));

    [C_d, corr_Y, d_opt_corr] = training_sync(Z_r, trainig_s, over, Ntr, d_total);

    h_hat = C_d / (Ntr*s_en);

    % the coherent |C_d| is attenuated by the phase drift 2*pi*dFs*over*(Ntr-1)
    % of the CFO, so we also synchronize with partial correlations over blocks
    % of Lb symbols
    Lb = 10;
    [corr_blk, d_opt_blk] = training_sync_blocks(Z_r, trainig_s, over, Ntr, d_total, Lb);

    Lplot = min(d_total, length(analog_channel));
    figure; hold on; grid on;
    plot(abs(h_hat(1:Lplot)), 'LineWidth', 1.2);
    plot(corr_blk(1:Lplot)/(Ntr*s_en), ':', 'LineWidth', 1.2);
    plot(abs(analog_channel(1:Lplot)), '--', 'LineWidth', 1.2);
    title('Training estimate vs |h_{composite}|, with CFO');
    xlabel('d (samples)'); ylabel('magnitude');
    legend('|C_d|/(N_{tr}\sigma_A^2)', 'block metric /(N_{tr}\sigma_A^2)', ...
           '|h_{composite}|', 'Location', 'best');
    saveas(gcf, fullfile(fig_dir, ['part2_2_3_training_vs_channel' tag '.png']));

    % theoretical attenuation of |C_d|, eq. (5.28):
    % |alpha| = |sin(pi*Df*Ntr)/(Ntr*sin(pi*Df))|
    alpha = abs(sin(pi*Df*Ntr)/(Ntr*sin(pi*Df)));
    fprintf('d_opt (energy) = %d, d_opt (training) = %d, d_opt (blocks) = %d\n', ...
            d_opt, d_opt_corr, d_opt_blk);
    fprintf('max|h_hat|/max|h| = %.4f (theory |alpha| = %.4f)\n', ...
            max(abs(h_hat(1:Lplot)))/max(abs(analog_channel(1:Lplot))), alpha);

    d_sync = d_opt_blk;

    idx_out = d_sync : over : d_sync + (N-1)*over;
    n_out = (idx_out - 1).';  % 0-based sample index of each symbol
    out_seq = Z_r(idx_out);

    figure;
    scatter(real(out_seq), imag(out_seq), 'filled'); grid on; axis equal;
    xlabel('Real'); ylabel('Imag');
    title('N symbol-spaced output sequence, CFO not corrected');
    saveas(gcf, fullfile(fig_dir, ['part2_2_3_constellation_cfo' tag '.png']));

    %2.5 CFO estimation with the training symbols
    % r_k = conj(A_k)*z_k has phase 2*pi*Df*k + theta0
    %   (a) coarse estimate with the FFT of r_k, Chapter 6
    %   (b) LS fit of the slope of the unwrapped phase

    r = out_seq(1:Ntr) .* conj(trainig_s);

    Nfft = 2^14;
    [Df_fft, f_axis, Zmag] = cfo_estimate_fft(out_seq(1:Ntr), trainig_s, Nfft);

    k_ax = (0:Ntr-1).';
    theta = unwrap(angle(r));
    P = [k_ax ones(Ntr,1)] \ theta;
    Df_hat = P(1) / (2*pi);        % cycles per symbol
    dFs_hat = Df_hat / over;       % cycles per sample

    fprintf('Df_hat (fft) = %.6f, Df_hat (LS) = %.6f (true %.6f)\n', ...
            Df_fft, Df_hat, Df);
    fprintf('dFs_hat = %.3e (true %.3e)\n', dFs_hat, dFs);

    figure;
    plot(f_axis, Zmag, 'LineWidth', 1.2); grid on; hold on;
    plot(Df_fft, max(Zmag), 'o', 'MarkerSize', 8, 'LineWidth', 1.5);
    xlim([-0.05 0.05]);
    xlabel('\Deltaf (cycles/symbol)'); ylabel('|Z(f)|');
    title(sprintf('Fourier transform of z_k = r_k A_k^*, \\Deltaf^* = %.5f', Df_fft));
    saveas(gcf, fullfile(fig_dir, ['part2_2_5_cfo_fft' tag '.png']));

    figure; hold on; grid on;
    plot(k_ax, theta, 'o');
    plot(k_ax, [k_ax ones(Ntr,1)]*P, 'LineWidth', 1.2);
    xlabel('k (symbols)'); ylabel('unwrapped phase of r_k (rad)');
    title('Phase drift of the training symbols and LS fit');
    legend('measured', 'LS fit', 'Location', 'best');
    saveas(gcf, fullfile(fig_dir, ['part2_2_5_phase_drift' tag '.png']));

    %2.6 CFO compensation

    out_corr = out_seq .* exp(-1i*2*pi*dFs_hat*n_out);

    figure;
    scatter(real(out_corr), imag(out_corr), 'filled'); grid on; axis equal;
    xlabel('Real'); ylabel('Imag');
    title('N symbol-spaced output sequence, CFO corrected');
    saveas(gcf, fullfile(fig_dir, ['part2_2_6_constellation_cfo_corrected' tag '.png']));

    %2.7 LS estimation of the discrete equivalent channel (one coefficient)
    % z_k = h*A_k -> h_hat = (A^H*z)/(A^H*A)

    Atr = trainig_s;
    ztr = out_corr(1:Ntr);

    h0_hat = (Atr' * ztr) / (Atr' * Atr);

    fprintf('h0_hat = %.4f %+.4fi, |h0_hat| = %.4f\n', ...
            real(h0_hat), imag(h0_hat), abs(h0_hat));
    fprintf('c      = %.4f %+.4fi, |c|      = %.4f\n', real(c), imag(c), abs(c));

    %2.8 channel inversion

    out_eq = out_corr / h0_hat;

    figure;
    scatter(real(out_eq), imag(out_eq), 'filled'); grid on; axis equal; hold on;
    scatter(real(packet), imag(packet), 60, 'x', 'LineWidth', 1.2);
    xlabel('Real'); ylabel('Imag');
    title('Output sequence after CFO correction and channel inversion');
    legend('received', 'transmitted 4-QAM', 'Location', 'best');
    saveas(gcf, fullfile(fig_dir, ['part2_2_8_constellation_equalized' tag '.png']));

    err = out_eq - packet;
    fprintf('max |z_eq - A| = %.3e, rms = %.3e\n', max(abs(err)), rms(err));

    dec = sign(real(out_eq)) + 1i*sign(imag(out_eq));
    fprintf('symbol errors = %d / %d\n', sum(dec ~= packet), N);
end
