% dOTF demo
clear; clc; close all;

%% grid
N = 1024;
[x, y] = meshgrid( (-N/2):(N/2-1), (-N/2):(N/2-1) );
r = sqrt(x.^2 + y.^2);

%% circular pupil
R_CS = 256;
CS = (r <= R_CS);

%% generate smooth random phase phi_0(x)
rng(1);

% white noise -> low-pass filtered in Fourier domain
noise_phase = randn(N, N);

[kx, ky] = meshgrid( (-N/2):(N/2-1), (-N/2):(N/2-1) );
kr = sqrt(kx.^2 + ky.^2);

sigma_k_phase = 5; % smaller -> smoother
LP_phase = exp(-(kr.^2) / (2*sigma_k_phase^2));

noise_phase_f = fftshift(fft2(noise_phase));
phi_0_smooth = real(ifft2(ifftshift(noise_phase_f .* LP_phase)));

% normalize and scale
phi_0_smooth = phi_0_smooth - mean(phi_0_smooth(CS));
phi_0_smooth = phi_0_smooth / std(phi_0_smooth(CS));

phi0_rms = 0.5; % radians RMS inside CS
phi_0 = zeros(N, N);
phi_0(CS) = phi0_rms * phi_0_smooth(CS);

%% generate slowly varying amplitude A_0(x)
noise_amp = randn(N, N);

sigma_k_amp = 10; % smoother than phase
LP_amp = exp(-(kr.^2) / (2*sigma_k_amp^2));

noise_amp_f = fftshift(fft2(noise_amp));
amp_smooth = real(ifft2(ifftshift(noise_amp_f .* LP_amp)));

amp_smooth = amp_smooth - mean(amp_smooth(CS));
amp_smooth = amp_smooth / std(amp_smooth(CS));

amp_variation = 0.08; % fractional variation
A_0 = zeros(N, N);
A_0(CS) = 1 + amp_variation * amp_smooth(CS);

% enforce positivity
A_0(CS) = max(A_0(CS), 0.05);

%% baseline pupil field
E_0 = A_0 .* exp(1i * phi_0);

%% focal-plane field and intensity
A = fftshift(fft2(ifftshift(E_0)));
I_0 = abs(A).^2;

%% quick diagnostics
fprintf('Grid size                : %d x %d\n', N, N);
fprintf('CS radius                : %d px\n', R_CS);
fprintf('CS pixels (N_CS)         : %d\n', nnz(CS));
fprintf('phi_0 RMS inside CS      : %.4f rad\n', std(phi_0(CS)));
fprintf('A_0 mean inside CS       : %.4f\n', mean(A_0(CS)));
fprintf('A_0 std inside CS        : %.4f\n', std(A_0(CS)));

%% plots
figure('Color','w');

subplot(2,3,1);
imagesc(CS); axis image off; colormap(gca, gray);
title('CS mask');

subplot(2,3,2);
imagesc(A_0); axis image off; colorbar;
title('A_0(x)');

subplot(2,3,3);
imagesc(phi_0); axis image off; colorbar;
title('\phi_0(x) [rad]');

subplot(2,3,4);
imagesc(real(E_0)); axis image off; colorbar;
title('Re\{E_0(x)\}');

subplot(2,3,5);
imagesc(imag(E_0)); axis image off; colorbar;
title('Im\{E_0(x)\}');

subplot(2,3,6);
imagesc(log10(I_0 + 1)); axis image off; colorbar;
title('log_{10}(I_0(\xi)+1)');



%% generate pupil with small obsucration
pupil_mask = ones(size(E_0));
pupil_mask(1:320,80+(end/2-10:end/2+10))=0;

E_1 = E_0.*pupil_mask;

figure;
subplot(1,2,1);imagesc(real(E_0));title('RE E_0');set(gca,'YDir','normal');
subplot(1,2,2);imagesc(real(E_1));title('RE E_1');set(gca,'YDir','normal');


%% get OTFs

O_0 = FH(abs(F(E_0)).^2);
O_1 = FH(abs(F(E_1)).^2);
mask_OTF = abs(O_1-O_0)>10;
dOTF = (O_1-O_0).*mask_OTF;
phi_dOTF = mod(angle(dOTF),2*pi)-pi;

c_lims =[min(phi_0(:)),max(phi_0(:))];

figure;
subplot(1,2,1);imagesc(phi_0);colorbar;clim(c_lims);set(gca,'YDir','normal');
subplot(1,2,2);imagesc(phi_dOTF);colorbar;clim(c_lims);set(gca,'YDir','normal');

%% helper functions

function U = F(u)
U = fftshift(fft2(ifftshift(u)));
end

function u = FH(U)
u = fftshift(ifft2(ifftshift(U))) * numel(U);
end