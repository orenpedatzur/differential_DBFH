function F = fft2c(x)
%FFT2C Centered, unitary 2D FFT.
F = fftshift(fft2(ifftshift(x))) / sqrt(numel(x));
end