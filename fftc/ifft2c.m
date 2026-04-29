function x = ifft2c(F)
%IFFT2C Centered, unitary inverse 2D FFT.
x = fftshift(ifft2(ifftshift(F))) * sqrt(numel(F));
end