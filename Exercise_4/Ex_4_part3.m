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

script_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(script_dir, 'figures', 'part3');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

%% 3
%3.1 (= step 1.1)

data_bits = randi([0 1], 2*(N-Ntr), 1); %300
training_bits = randi([0 1], 2*Ntr, 1); %100

g(:,1) = srrc_pulse(T,Ts,B,beta);

data_s = bits_to_4qam(data_bits);    %150
trainig_s = bits_to_4qam(training_bits); %50

packet = [trainig_s;data_s]; % 200, First 50 training symbols

packet_up = upsample(packet,over); %for correct conv later

A = conv(packet_up,g); % low pass equivalent of channel input

%3.2 non ideal physical channel c(t) = c0*delta(t) + c1*delta(t - K*Ts)

K = randi([1 4*over]);   % delay of the second path, in samples
rho = 0.6;               % relative strength of the second path
c0 = (randn + 1i*randn)/sqrt(2);
c1 = rho*(randn + 1i*randn)/sqrt(2);

c = zeros(K+1,1);
c(1) = c0;
c(K+1) = c1;

fprintf('K = %d samples, |c0| = %.3f, |c1| = %.3f\n', K, abs(c0), abs(c1));

Y_r = conv(A,c); %received from channel (noiseless)

%3.3 (= steps 1.3 and 1.4)

Z_r = conv(Y_r,g)*Ts; %filtered

analog_channel = conv(conv(c,g),g)*Ts;
span = length(analog_channel);   % 4*B*over + 1 + K

x = span - 1;
figure;
plot(0:Ts:x*Ts,abs(analog_channel));
xlabel("Time");
ylabel("Amplitude")
grid on;
title("Composite Analog Channel (non ideal)");
saveas(gcf, fullfile(fig_dir, 'part3_3_3_analog_channel.png'));

figure; hold on; grid on;
plot(0:Ts:x*Ts, real(analog_channel), 'LineWidth', 1.2);
plot(0:Ts:x*Ts, imag(analog_channel), '--', 'LineWidth', 1.2);
xlabel("Time"); ylabel("Amplitude");
title("Composite Analog Channel, real and imaginary part");
legend('Re\{h(t)\}', 'Im\{h(t)\}', 'Location', 'best');
saveas(gcf, fullfile(fig_dir, 'part3_3_3_analog_channel_re_im.png'));

%3.4 (= step 1.6) corr_d for d = 1,...,4*B*over + 1 + K

d_total_corr = span;

[C_d, corr_Y, d_opt_corr] = training_sync(Z_r, trainig_s, over, Ntr, d_total_corr);

h_hat = C_d / (Ntr*s_en); % estimate of the sampled composite channel

figure; hold on; grid on;
plot(corr_Y/max(corr_Y), 'LineWidth', 1.2);
plot(abs(analog_channel)/max(abs(analog_channel)), '--', 'LineWidth', 1.2);
title('corr_d vs |h_{composite}| (norm), non ideal channel');
xlabel('d (samples)'); ylabel('normalized');
legend('|C_d| (training metric)', '|h_{composite}|', 'Location', 'best');
saveas(gcf, fullfile(fig_dir, 'part3_3_4_corr_vs_channel.png'));

%3.5 E_d assuming discrete equivalent channel of length M
% the M taps are spaced by over samples, so the window (M-1)*over must fit
% inside the composite channel -> M = floor((span-1)/over) and
% d = 1,...,span - (M-1)*over (one symbol period of timing phases)

M = floor((span-1)/over);
d_total = span - (M-1)*over;

fprintf('span = %d samples, M = %d taps, d = 1,...,%d\n', span, M, d_total);

[Ed, d_sync] = energy_Synchronization(h_hat, over, M, d_total);

figure;
plot(1:length(Ed), Ed); grid on;
title('E_d from h_{hat}, window of M taps');
xlabel('d (samples)'); ylabel('E_d');
saveas(gcf, fullfile(fig_dir, 'part3_3_5_energy_stat.png'));

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
title('Symbol-spaced output of the non ideal channel (before equalization)');
saveas(gcf, fullfile(fig_dir, 'part3_3_5_constellation_unequalized.png'));

% the discrete equivalent channel is the composite channel sampled every T,
% starting from the synchronization instant
analog_pad = [analog_channel; complex(zeros(d_sync + (M-1)*over, 1))];
h_disc = analog_pad(d_sync : over : d_sync + (M-1)*over);

figure; hold on; grid on;
plot((0:span-1)*Ts, abs(analog_channel), 'LineWidth', 1.2);
stem((d_sync-1 + (0:M-1)*over)*Ts, abs(h_disc), 'filled');
xlabel('t / T'); ylabel('magnitude');
title('Composite analog and discrete equivalent channel');
legend('|h(t)|', '|h_m|, discrete equivalent', 'Location', 'best');
saveas(gcf, fullfile(fig_dir, 'part3_3_5_discrete_channel.png'));

%3.5b LS estimation of the discrete equivalent channel of length M
% y_n = sum_{m=0}^{M-1} h_m*A_{n-m} -> use the rows that only contain training

h_ls = ls_channel_estimate(Y, trainig_s, M);

% true discrete equivalent channel, for comparison
analog_pad = [analog_channel; complex(zeros(need_len,1))];
h_true = analog_pad(d_sync : over : d_sync + (M-1)*over);

train_matr = training_matrix(trainig_s, M);
y_ls = Y(M:Ntr);
fprintf('LS residual energy = %.3e\n', sum(abs(y_ls - train_matr*h_ls).^2));
fprintf('||h_ls - h_true|| / ||h_true|| = %.3e\n', norm(h_ls - h_true)/norm(h_true));

figure;
subplot(2,1,1); hold on; grid on;
stem(0:M-1, real(h_true), 'o', 'LineWidth', 1.2);
stem(0:M-1, real(h_ls), 'x--', 'LineWidth', 1.2);
xlabel('m'); ylabel('Re\{h_m\}'); title('Real part');
legend('true', 'LS estimate', 'Location', 'best');
subplot(2,1,2); hold on; grid on;
stem(0:M-1, imag(h_true), 'o', 'LineWidth', 1.2);
stem(0:M-1, imag(h_ls), 'x--', 'LineWidth', 1.2);
xlabel('m'); ylabel('Im\{h_m\}'); title('Imaginary part');
legend('true', 'LS estimate', 'Location', 'best');
saveas(gcf, fullfile(fig_dir, 'part3_3_5_channel_estimate.png'));

%3.6 ZF equalizer of length L, for delay delta

L = 5*M;
delta = M;

w = zf_equalizer(h_ls, L, delta);   % LS solution of H*w = e_delta

comb = conv(h_ls,w); % combined channel + equalizer response

fprintf('combined response: |g_delta| = %.4f, max |g_n| (n ~= delta) = %.3e\n', ...
        abs(comb(delta+1)), max(abs(comb([1:delta delta+2:end]))));

figure;
stem(0:length(comb)-1, abs(comb), 'LineWidth', 1.2); grid on;
xlabel('n'); ylabel('|(h * w)_n|');
title(sprintf('Combined response channel - ZF equalizer, \\delta = %d', delta));
saveas(gcf, fullfile(fig_dir, 'part3_3_6_combined_response.png'));

%3.7 equalize, drop the first delta samples, keep the next N

Y_eq = conv(Y,w);
out_eq = Y_eq(delta+1 : delta+N);

figure;
scatter(real(out_eq), imag(out_eq), 'filled'); grid on; axis equal; hold on;
scatter(real(packet), imag(packet), 60, 'x', 'LineWidth', 1.2);
xlabel('Real'); ylabel('Imag');
title('Output sequence after the ZF equalizer');
legend('equalized', 'transmitted 4-QAM', 'Location', 'best');
saveas(gcf, fullfile(fig_dir, 'part3_3_7_constellation_equalized.png'));

err = out_eq - packet;
fprintf('max |z_eq - A| = %.3e, rms = %.3e\n', max(abs(err)), rms(err));

dec = sign(real(out_eq)) + 1i*sign(imag(out_eq));
fprintf('symbol errors = %d / %d\n', sum(dec ~= packet), N);
