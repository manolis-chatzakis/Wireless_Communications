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
fig_dir = fullfile(script_dir, 'figures', 'part1');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

%% 1
%1.1

data_bits = randi([0 1], 2*(N-Ntr), 1); %300
training_bits = randi([0 1], 2*Ntr, 1); %100

g(:,1) = srrc_pulse(T,Ts,B,beta);

data_s = bits_to_4qam(data_bits);    %150
trainig_s = bits_to_4qam(training_bits); %50

packet = [trainig_s;data_s]; % 200 symbols, the first 50 are training

packet_up = upsample(packet,over); %for correct conv later

A = conv(packet_up,g); % low pass equivalent of channel input

figure; hold on; grid on;
t_axis = (0:length(A)-1)*Ts;
plot(t_axis, real(A), 'LineWidth', 1.2);
plot(t_axis, imag(A), 'LineWidth', 1.2);
xlim([0 30]);
xlabel('t / T'); ylabel('A(t)');
title('Low-pass equivalent channel input A(t)');
legend('Re\{A(t)\}', 'Im\{A(t)\}', 'Location', 'best');
saveas(gcf, fullfile(fig_dir, 'part1_1_1_channel_input.png'));

%% channel cases
% 1.2 - 1.7 : c(t) = delta(t)            -> c = 1
% 1.8       : c(t) = c*delta(t), c in C  -> c complex

c_rand = (randn + 1i*randn)/sqrt(2);

c_values = {1, c_rand};
tags = {'_ideal', '_complex'};

for ic = 1:numel(c_values)

    c = c_values{ic};
    tag = tags{ic};

    fprintf('\n--- case %s: c = %.4f %+.4fi ---\n', tag, real(c), imag(c));

    % 1.2 (noiseless channel, c(t) = c*delta(t))

    Y_r = conv(A,c); %received from channel

    %1.3
    Z_r = conv(Y_r,g)*Ts; %filtered

    %1.4
    analog_channel = conv(conv(c,g),g)*Ts;
    x = length(analog_channel) - 1;
    figure;
    plot(0:Ts:x*Ts,abs(analog_channel));
    xlabel("Time");
    ylabel("Amplitude")
    grid on;
    title("Composite Analog Channel");
    saveas(gcf, fullfile(fig_dir, ['part1_1_4_analog_channel' tag '.png']));

    %1.5
    % search range given by the exercise: d = 0,...,4*B*over - 1
    % (1-based MATLAB indices: d = 1,...,4*B*over)
    d_total = 4*B*over;

    [Ed,d_opt] = energy_Synchronization(Z_r, over, N, d_total);

    figure;
    plot(1:length(Ed), Ed); grid on;
    title('E_d ');
    xlabel('d (samples)'); ylabel('E_d');
    saveas(gcf, fullfile(fig_dir, ['part1_1_5_energy_stat' tag '.png']));

    % 1.6

    [C_d, corr_Y, d_opt_corr] = training_sync(Z_r, trainig_s, over, Ntr, d_total);

    h_hat = C_d / (Ntr*s_en); % theory eq 5.20: C_d/(Ntr*sigma_A^2), not squared

    % Plot: |h_hat| = |C_d|/(Ntr*sigma_A^2) vs |analog_channel| (composite)
    Lplot = min(d_total, length(analog_channel));
    figure; hold on; grid on;
    plot(abs(h_hat(1:Lplot)), 'LineWidth', 1.2);
    plot(abs(analog_channel(1:Lplot)), '--', 'LineWidth', 1.2);
    title('Training estimate vs |h_{composite}|');
    xlabel('d (samples)'); ylabel('magnitude');
    legend('|C_d|/(N_{tr}\sigma_A^2)', '|h_{composite}|', 'Location', 'best');
    saveas(gcf, fullfile(fig_dir, ['part1_1_6_training_vs_channel' tag '.png']));

    fprintf('d_opt (energy) = %d, d_opt (training) = %d\n', d_opt, d_opt_corr);
    fprintf('max|h_hat| / max|h| = %.4f\n', ...
            max(abs(h_hat(1:Lplot))) / max(abs(analog_channel(1:Lplot))));

    %1.7

    out_seq = Z_r(d_opt_corr : over : d_opt_corr + (N-1)*over);

    figure;
    scatter(real(out_seq), imag(out_seq), 'filled'); grid on; axis equal;
    xlabel('Real'); ylabel('Imag');
    title('N symbol-spaced output sequence (training-based sync)');
    saveas(gcf, fullfile(fig_dir, ['part1_1_7_constellation' tag '.png']));

    err = out_seq - c*packet;
    fprintf('max |z_k - c*A_k| = %.3e\n', max(abs(err)));

    %1.8
    % the second pass of the loop is step 1.8, with c complex
end
