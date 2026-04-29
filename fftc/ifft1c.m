function x = ifft1c(F)
%IFFT1C Centered 1D inverse FFT. Non-unitary to match the KK prototype.
x = fftshift(ifft(ifftshift(F)));
end