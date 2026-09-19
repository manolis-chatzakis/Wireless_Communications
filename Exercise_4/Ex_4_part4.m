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
phi = 2*pi*rand; % random carrier phase

script_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(script_dir, 'figures', 'part4');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

%% 4
%4.1 same steps as in part 3, but now the received signal also has CFO

data_bits = randi([0 1], 2*(N-Ntr), 1); %300
training_bits = randi([0 1], 2*Ntr, 1); %100

g(:,1) = srrc_pulse(T,Ts,B,beta);

data_s = bits_to_4qam(data_bits);    %150
trainig_s = bits_to_4qam(training_bits); %50

packet = [trainig_s;data_s]; % 200, First 50 training symbols

packet_up = upsample(packet,over); %for correct conv later

A = conv(packet_up,g); % low pass equivalent of channel input

%4.2 non ideal physical channel c(t) = c0*delta(t) + c1*delta(t - K*Ts)

K = randi([1 4*over]);   % delay of the second path, in samples
rho = 0.6;               % relative strength of the second path
c0 = (randn + 1i*randn)/sqrt(2);
c1 = rho*(randn + 1i*randn)/sqrt(2);

c = zeros(K+1,1);
c(1) = c0;
c(K+1) = c1;

fprintf('K = %d samples, |c0| = %.3f, |c1| = %.3f\n', K, abs(c0), abs(c1));
fprintf('dFs = %.4g (Df = %.4g), phi = %.4f rad\n', dFs, Df, phi);

Y_r = conv(A,c); %received from channel (noiseless)

% CFO: multiply by exp(j*(2*pi*dFs*n + phi)), n = t/Ts
n_ax = (0:length(Y_r)-1).';
Y_cfo = Y_r .* exp(1i*(2*pi*dFs*n_ax + phi));

%4.3 matched filter and composite analog channel (= steps 1.3 and 1.4)

Z_r = conv(Y_cfo,g)*Ts; %filtered

% the CFO has unit modulus, so the magnitude of the composite channel is the
% same as in part 3
analog_channel = conv(conv(c,g),g)*Ts;
span = length(analog_channel);   % 4*B*over + 1 + K

x = span - 1;
figure;
plot(0:Ts:x*Ts,abs(analog_channel));
xlabel("Time");
ylabel("Amplitude")
grid on;
title("Composite Analog Channel (non ideal, with CFO)");
saveas(gcf, fullfile(fig_dir, 'part4_4_3_analog_channel.png'));

%4.4 (= step 3.4) corr_d for d = 1,...,4*B*over + 1 + K

d_total_corr = span;

[C_d, corr_Y, d_opt_corr] = training_sync(Z_r, trainig_s, over, Ntr, d_total_corr);

h_hat = C_d / (Ntr*s_en); % estimate of the sampled composite channel

figure; hold on; grid on;
plot(abs(h_hat), 'LineWidth', 1.2);
plot(abs(analog_channel), '--', 'LineWidth', 1.2);
title('corr_d vs |h_{composite}|, non ideal channel with CFO');
xlabel('d (samples)'); ylabel('magnitude');
legend('|C_d|/(N_{tr}\sigma_A^2)', '|h_{composite}|', 'Location', 'best');
saveas(gcf, fullfile(fig_dir, 'part4_4_4_training_vs_channel.png'));

% the CFO attenuates the coherent correlation by |alpha|, eq. (5.28)
alpha = abs(sin(pi*Df*Ntr)/(Ntr*sin(pi*Df)));
fprintf('max|h_hat| / max|h| = %.4f (theory |alpha| = %.4f)\n', ...
        max(abs(h_hat))/max(abs(analog_channel)), alpha);

%4.5 (= step 3.5) E_d assuming discrete equivalent channel of length M
% the attenuation |alpha| of the CFO is common to all d, so E_d can still be
% used for the timing synchronization

M = floor((span-1)/over);
d_total = span - (M-1)*over;

fprintf('span = %d samples, M = %d taps, d = 1,...,%d\n', span, M, d_total);

[Ed, d_sync] = energy_Synchronization(h_hat, over, M, d_total);

figure;
plot(1:length(Ed), Ed); grid on;
title('E_d from h_{hat} (with CFO), window of M taps');
xlabel('d (samples)'); ylabel('E_d');
saveas(gcf, fullfile(fig_dir, 'part4_4_5_energy_stat.png'));

fprintf('d_sync = %d, d_opt (corr) = %d\n', d_sync, d_opt_corr);

% symbol-spaced output sequence of length N+M-1
need_len = d_sync + (N+M-2)*over;
if length(Z_r) < need_len
    Z_r = [Z_r; complex(zeros(need_len - length(Z_r),1))];
end
Y = Z_r(d_sync : over : d_sync + (N+M-2)*over);

figure;
scatter(real(Y), imag(Y), 'filled'); grid on; axis equal;
xlabel('Real'); ylabel('Imag');
title('Symbol-spaced output, non ideal channel with CFO');
saveas(gcf, fullfile(fig_dir, 'part4_4_5_constellation_cfo.png'));

% what happens if the CFO is ignored and part 3 is applied directly
h_ls_noCFO = ls_channel_estimate(Y, trainig_s, M);
w_noCFO = zf_equalizer(h_ls_noCFO, 5*M, M);
z_noCFO = conv(Y, w_noCFO);
z_noCFO = z_noCFO(M+1 : M+N);
dec_noCFO = sign(real(z_noCFO)) + 1i*sign(imag(z_noCFO));
fprintf('without CFO correction: symbol errors = %d / %d\n', ...
        sum(dec_noCFO ~= packet), N);

figure;
scatter(real(z_noCFO), imag(z_noCFO), 'filled'); grid on; axis equal;
xlabel('Real'); ylabel('Imag');
title('ZF output without CFO correction');
saveas(gcf, fullfile(fig_dir, 'part4_4_5_equalized_noCFO.png'));

%4.6 joint LS estimation of the CFO and of the discrete equivalent channel
% (Section 5.3.1): Df* = argmax y^H*Gamma(Df)*P*Gamma(Df)^H*y

df_grid = -0.5 : 1e-5 : 0.5-1e-5;
[Df_hat, h_star, cost] = joint_cfo_channel_ls(Y, trainig_s, M, df_grid);

fprintf('Df_hat = %.5f (true %.5f), dFs_hat = %.3e (true %.3e)\n', ...
        Df_hat, Df, Df_hat/over, dFs);

figure;
plot(df_grid, cost, 'LineWidth', 1.2); hold on; grid on;
plot(Df_hat, max(cost), 'o', 'MarkerSize', 8, 'LineWidth', 1.5);
xlim([-0.05 0.05]);
xlabel('\Deltaf (cycles/symbol)'); ylabel('g(\Deltaf)');
title(sprintf('Joint LS cost, \\Deltaf^* = %.5f', Df_hat));
saveas(gcf, fullfile(fig_dir, 'part4_4_6_cfo_cost.png'));

% CFO correction, eq. (5.42)
k_ax = (0:N+M-2).';
Y_corr = Y .* exp(-1i*2*pi*Df_hat*k_ax);

figure;
scatter(real(Y_corr), imag(Y_corr), 'filled'); grid on; axis equal;
xlabel('Real'); ylabel('Imag');
title('Symbol-spaced output after CFO correction');
saveas(gcf, fullfile(fig_dir, 'part4_4_6_constellation_cfo_corrected.png'));

% LS estimation of the discrete channel on the corrected sequence
h_ls = ls_channel_estimate(Y_corr, trainig_s, M);
fprintf('||h_ls - h_star|| = %.3e (LS after correction vs joint LS)\n', ...
        norm(h_ls - h_star));

% true discrete equivalent channel (without CFO), for comparison
analog_pad = [analog_channel; complex(zeros(d_sync + (M-1)*over, 1))];
h_disc = analog_pad(d_sync : over : d_sync + (M-1)*over);

figure; hold on; grid on;
stem(0:M-1, abs(h_disc), 'o', 'LineWidth', 1.2);
stem(0:M-1, abs(h_ls), 'x--', 'LineWidth', 1.2);
xlabel('m'); ylabel('|h_m|');
title('Magnitude of the discrete equivalent channel');
legend('true (without CFO)', 'LS estimate (CFO corrected)', 'Location', 'best');
saveas(gcf, fullfile(fig_dir, 'part4_4_6_ls_channel.png'));

%4.7 ZF equalizer of length L, for delay delta

L = 5*M;
delta = M;

w = zf_equalizer(h_ls, L, delta);

comb = conv(h_ls,w); % combined channel + equalizer response

fprintf('combined response: |g_delta| = %.4f, max |g_n| (n ~= delta) = %.3e\n', ...
        abs(comb(delta+1)), max(abs(comb([1:delta delta+2:end]))));

figure;
stem(0:length(comb)-1, abs(comb), 'LineWidth', 1.2); grid on;
xlabel('n'); ylabel('|(h * w)_n|');
title(sprintf('Combined response channel - ZF equalizer (CFO), \\delta = %d', delta));
saveas(gcf, fullfile(fig_dir, 'part4_4_7_combined_response.png'));

% equalize, drop the first delta samples, keep the next N
Y_eq = conv(Y_corr,w);
out_eq = Y_eq(delta+1 : delta+N);

figure;
scatter(real(out_eq), imag(out_eq), 'filled'); grid on; axis equal; hold on;
scatter(real(packet), imag(packet), 60, 'x', 'LineWidth', 1.2);
xlabel('Real'); ylabel('Imag');
title('Output sequence after CFO correction and ZF equalization');
legend('equalized', 'transmitted 4-QAM', 'Location', 'best');
saveas(gcf, fullfile(fig_dir, 'part4_4_7_constellation_equalized.png'));

err = out_eq - packet;
fprintf('max |z_eq - A| = %.3e, rms = %.3e\n', max(abs(err)), rms(err));

dec = sign(real(out_eq)) + 1i*sign(imag(out_eq));
fprintf('symbol errors = %d / %d\n', sum(dec ~= packet), N);
