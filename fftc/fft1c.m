function F = fft1c(x)
%FFT1C Centered 1D FFT. Non-unitary to match the KK prototype.
F = fftshift(fft(ifftshift(x)));
end