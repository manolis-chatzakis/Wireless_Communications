function [Df_hat, f_axis, Zmag] = cfo_estimate_fft(r_tr, train, Nfft)
% CFO estimation with training symbols in an ideal channel (Chapter 6)
%
% r_tr  : symbol-spaced outputs that correspond to the training symbols
%         r_l = h * a_l * exp(j(2*pi*Df*l + phi)),  l = 0,...,Ntr-1
% train : the training symbols a_l (known at the receiver)
% Nfft  : number of FFT points (zero padding -> fine frequency grid)
%
% Df_hat : estimate of Df = DF*T (cycles per symbol)

% eq. (6.16): remove the symbols -> complex exponential with frequency Df
z = r_tr .* conj(train);

% Fourier transform of z_l; the peak gives the frequency of the exponential
Z = fftshift(fft(z, Nfft));
f_axis = (-Nfft/2 : Nfft/2-1) / Nfft;     % digital frequencies in [-1/2, 1/2)
Zmag = abs(Z);

[~, i_max] = max(Zmag);
Df_hat = f_axis(i_max);
end
