%% demo_differential_DBFH.m
% demo for full process:
% -->recovering Im(A^* deltaA) from Re(A^* deltaA)
% -->recovering focal plane phasors X
%
% Geometry:
%   supp_large = large circular aperture
%   supp_small = smaller circular aperture fully inside large
%   supp1 = large minus small
%   supp2 = overlap region (small patch inside large)
%
% Field model:
%   one parent field E on supp_large
%   E1 = E .* supp1
%   E2 = E .* supp2
%
% Cross-term:
%   z = A1 .* conj(A2)

clear; close all; clc;

addpath('diff_DBFH\');
addpath('sign_problem\');

parula_with_nan_white = [1 1 1; parula(256)];

%% ----------- SIGN PROBLEM --------
%% ---------- parameters ----------
tic;

N  = 2^10;
R1 = 0.25;
R2 = 0.05;

x1 = 0.00; y1 = 0.00;   % large aperture center

d12 = R1;%-1.1*R2;
theta2 = deg2rad(30);
x2 = d12*cos(theta2); y2 = d12*sin(theta2);   % small aperture center (fully inside)

phase_rms = 1.0;
amp_jitter = 0.15;
sigma_px = 10;

rng(1);

%% ---------- centered FFT helpers ----------
fft2c  = @(u) fftshift(fft2(ifftshift(u))) / N;
ifft2c = @(u) fftshift(ifft2(ifftshift(u))) * N;

%% ---------- grid ----------
[x, y] = meshgrid(linspace(-1,1,N), linspace(-1,1,N));

%% ---------- supports ----------
supp_large = ((x - x1).^2 + (y - y1).^2) <= R1^2;
supp_small = ((x - x2).^2 + (y - y2).^2) <= R2^2;

% if any(supp_small(:) & ~supp_large(:))
%     error('Small aperture must be fully inside the large aperture.');
% end

supp1 = supp_large & ~supp_small;
supp2 = supp_large &  supp_small;

%% ---------- one parent field ----------
phi = imgaussfilt(randn(N), sigma_px);
phi = phi - mean(phi(supp_large));
phi = phase_rms * phi / std(phi(supp_large));

amp = 1 + amp_jitter * imgaussfilt(randn(N), sigma_px);
amp(amp < 0) = 0;

E = supp_large .* amp .* exp(1i * phi);
E1 = E .* supp1;
E2 = E .* supp2;
deltaE = E2;

figure;
subplot(1,2,1);imagesc(imfuse(real(E),supp_large-supp_small));axis equal;title('Re(E)');
subplot(1,2,2);imagesc(imfuse(imag(E),supp_large-supp_small));axis equal;title('Re(E)');

%% ---------- Fourier-domain fields ----------
A1 = fft2c(E1);
A2 = fft2c(E2);
A  = A1 + A2;

I1 = abs(A1).^2;
I  = abs(A).^2;

% focal-plane phasor truth - for comparison with estimate only
X_true = A ./ max(abs(A), eps);


figure;
subplot(1,2,1);imagesc(real(E).*supp1);axis equal tight;title('Re(E) cookie bite');

g = ifft2c(conj(A1).*A2);
subplot(1,2,2);imagesc(abs(g));title('g(x)=IFT[A_1^*A_2]');axis equal tight;

%%
supp1 = supp_large & ~(x >= 0.9*R1);
supp2 = supp_large & (x >= 0.9*R1);


CSg = conv2(double(flipud(fliplr(supp1))), double(supp2), 'same') > 0;
figure;
subplot(1,3,3);imagesc(CSg);hold on;plot(N/2,N/2,'*k');axis equal tight;title('support of g');
subplot(1,3,1);imagesc(supp1);hold on;plot(N/2,N/2,'*k');axis equal tight;title('support 1');
subplot(1,3,2);imagesc(supp2);hold on;plot(N/2,N/2,'*k');axis equal tight;title('support 2');

%% ---------- cross term ----------
z_true   = A1 .* conj(A2);
rez      = real(z_true);
absz     = abs(z_true);
imz_true = imag(z_true);
c_true = ifft2c(z_true);

%% ---------- consistency check from measurements ----------
A2_mag_sq_from_meas = I - I1 - 2*rez;
A2_mag_sq_from_meas(A2_mag_sq_from_meas < 0) = 0;

rel_err_A2sq = norm(A2_mag_sq_from_meas - abs(A2).^2, 'fro') / ...
               max(norm(abs(A2).^2, 'fro'), eps);

fprintf('relative error in |A2|^2 from measurements: %.3e\n', rel_err_A2sq);

%% ---------- support mask for ifft2(z) ----------

% support of ifft2(A1 .* conj(A2))) from supports alone
corr_support = ifft2c( fft2c(double(supp1)) .* conj(fft2c(double(supp2))) );
support_mask = abs(corr_support) > 1e-15 * max(abs(corr_support(:)));

%% ---------- solve sign problem ----------
[out_sign_problem, seg_id] = solve_sign_problem_with_segments( ...
    absz, rez, support_mask, ...
    'SmoothSigma', 0.3, ...
    'QThresholdRel', 1e-4, ...
    'MinSegmentPixels', 4, ...
    'Tol', 1e-10, ...
    'MaxIter', 1e3, ...
    'LambdaSeg', 1e0, ...
    'Verbose', true, ...
    'UseSingle', false,...
    'ShowPlots',true);

z_est = out_sign_problem.z_sign;
imz_est = imag(out_sign_problem.z_sign);
c_est   = ifft2c(out_sign_problem.z_sign); % TODO : this is not C(\xi)...

%% ---------- diagnostics ----------
rel_err_imz = norm(imz_est - imz_true, 'fro') / max(norm(imz_true, 'fro'), eps);
rel_err_z   = norm(z_est   - z_true,  'fro') / max(norm(z_true,  'fro'), eps);

leak_true = norm(c_true(~support_mask)) / max(norm(c_true(:)), eps);
leak_est  = norm(c_est (~support_mask)) / max(norm(c_est(:)),  eps);

fprintf('relative error in Im(z): %.3e\n', rel_err_imz);
fprintf('relative error in z:     %.3e\n', rel_err_z);
fprintf('true leakage outside support: %.3e\n', leak_true);
fprintf('est  leakage outside support: %.3e\n', leak_est);

%% ---------- plots ----------
figure;

subplot(3,4,1);
imagesc(supp_large + 2*supp_small); axis image off; colorbar;
title('large + small supports');

subplot(3,4,2);
imagesc(abs(E).*supp_large); axis image off; colorbar;
title('|E|');

subplot(3,4,3);
imagesc(real(E).*supp_large); axis image off; colorbar;
title('real(E)');

subplot(3,4,4);
imagesc(imag(E).*supp_large); axis image off; colorbar;
title('imag(E)');

axf(1) = subplot(3,4,5);
imagesc(rez); axis image off; colorbar;
title('Re(z)');

axf(2) = subplot(3,4,6);
imagesc(absz); axis image off; colorbar;
title('|z|');

axf(3) = subplot(3,4,7);
imagesc(out_sign_problem.q); axis image off; colorbar;
title('q=|Im(z)|');

axf(4) = subplot(3,4,8);
imagesc(imfuse(seg_id==0,out_sign_problem.q)); axis image off; colorbar;
title('q=|Im(z)| + segmentation');

axf(5) = subplot(3,4,9);
imagesc(imz_true); axis image off; colorbar;
title('Im(z) true');

axf(6) = subplot(3,4,10);
imagesc(imz_est); axis image off; colorbar;
title(sprintf('Im(z) est, err=%.2e', rel_err_imz));

axf(7) = subplot(3,4,11);
imagesc(abs(imz_est-imz_true)); axis image off; colorbar;
title('Im(z) (estimated) - (true)');

linkaxes(axf);

sgtitle('Sign-recovery demo for z = A conj(\delta A)');

%% ---------- some rows comparison ----------
row_inds = round(N/2)+round(N/100)*[-3:1:3];
plot_y_scale = max(abs(imz_true(:)));
figure;
for ind_subplot = 1:numel(row_inds)
    subplot(numel(row_inds),1,ind_subplot);
    plot(imz_true(row_inds(ind_subplot),:), 'k-', 'LineWidth', 1.5); hold on;
    plot(imz_est(row_inds(ind_subplot),:),  'r--', 'LineWidth', 1.2);
    grid on;
    legend('true','estimated');
    title(['Im(z) selected row:',num2str(row_inds(ind_subplot))]);
    xlabel('column');
    ylabel('Im(z)');
    ylim(plot_y_scale*[-1,1]);
end

%% ---------- support-domain comparison ----------
figure;

ax1 = subplot(1,3,1);
imagesc(abs(c_true)); axis image off; colorbar;
title('c true = |ifft2(z true)|');

ax2 = subplot(1,3,2);
imagesc(abs(c_est)); axis image off; colorbar;
title('c est = |ifft2(z est)|');

subplot(1,3,3);
imagesc(abs(c_est - c_true)); axis image off; colorbar;
title('|difference|');

linkaxes([ax1,ax2]);

%% ----------- diff DBFH -----------
%% run diff_DBFH
[~,anchor_idx] = max(I(:));

out_DBFH = diff_DBFH( ...
    I, conj(z_est), supp_large, supp_small, ...
    'KRelThresh',1e-4,...
    'LambdaCS', 1, ...
    'LambdaRef', 1, ...
    'AnchorIndex', anchor_idx, ...
    'LSQRTol', 1e-10, ...
    'LSQRMaxIter', 3e3, ...
    'Verbose', true,...
    'ShowPlots',true);


X_rec = out_DBFH.X_norm;

%% align global phase
alpha = sum(conj(X_rec(:)) .* X_true(:));
alpha = alpha / max(abs(alpha), eps);
X_rec_aligned = X_rec * alpha;

%% errors
rel_err_X = norm(X_rec_aligned(:) - X_true(:)) / max(norm(X_true(:)), eps);

phase_err = angle(exp(1i * (angle(X_rec_aligned) - angle(X_true))));

w = I./sum(I,'all');

rmse_phase = sqrt(sum(w.*phase_err.^2,'all'));

fprintf('\nAfter global phase alignment:\n');
fprintf('  rel error X   = %.3e\n', rel_err_X);
fprintf('  weighted phase RMSE    = %.3e rad\n', rmse_phase);

%% plots: pupil fields
figure;

subplot(3,3,1);
imagesc(abs(E)); axis image; colorbar;
title('|E true|');

subplot(3,3,2);
imagesc(angle(E)); axis image; colorbar;
title('angle(E true)');

subplot(3,3,3);
imagesc(supp_large); axis image; colorbar;
title('CS E');

subplot(3,3,4);
imagesc(abs(deltaE)); axis image; colorbar;
title('|deltaE true|');

subplot(3,3,5);
imagesc(angle(deltaE)); axis image; colorbar;
title('angle(deltaE true)');

subplot(3,3,6);
imagesc(supp_large); axis image; colorbar;
title('CS E');

subplot(3,3,7);
imagesc(log10(I + 1e-12)); axis image; colorbar;
title('log10(I full)');

subplot(3,3,8);
imagesc(real(c_est)); axis image; colorbar;
title('Re(C_{est})');

subplot(3,3,9);
imagesc(imag(c_est)); axis image; colorbar;
title('Im(C_{est})');

colormap parula;

%%  plots: recovered focal-plane phasor + differences
figure();

% compute differences
dRe   = real(X_rec_aligned) - real(X_true);
dIm   = imag(X_rec_aligned) - imag(X_true);
dPhi  = angle(exp(1i*(angle(X_rec_aligned) - angle(X_true)))); % wrapped

ax = gobjects(9,1);

% --- Row 1: Real ---
ax(1) = subplot(3,3,1);
imagesc(real(X_rec_aligned)); axis image; colorbar;
title('Re(X) recovered');

ax(2) = subplot(3,3,2);
imagesc(real(X_true)); axis image; colorbar;
title('Re(X) true');

ax(3) = subplot(3,3,3);
imagesc(dRe); axis image; colorbar;
title('\Delta Re');

% --- Row 2: Imag ---
ax(4) = subplot(3,3,4);
imagesc(imag(X_rec_aligned)); axis image; colorbar;
title('Im(X) recovered');

ax(5) = subplot(3,3,5);
imagesc(imag(X_true)); axis image; colorbar;
title('Im(X) true');

ax(6) = subplot(3,3,6);
imagesc(dIm); axis image; colorbar;
title('\Delta Im');

% --- Row 3: Phase ---
ax(7) = subplot(3,3,7);
imagesc(angle(X_rec_aligned)); axis image; colorbar;
title('\phi recovered');

ax(8) = subplot(3,3,8);
imagesc(angle(X_true)); axis image; colorbar;
title('\phi true');

ax(9) = subplot(3,3,9);
imagesc(dPhi); axis image; colorbar;
title('\Delta \phi (wrapped)');

colormap parula;

% link all axes
linkaxes(ax, 'xy');


%% plots: pupil plane comparison |E| and phase
E_rec = out_DBFH.E;

% align global phase in pupil (same alpha as focal plane is fine)
alpha_E = exp(1i * angle(sum(conj(E_rec(supp_large)) .* E(supp_large))));
E_rec_aligned = E_rec * alpha_E;

% differences
dAbsE = abs(E_rec_aligned) - abs(E);
dPhiE = angle(exp(1i*(angle(E_rec_aligned) - angle(E))));

figure;
supp_large_nan = double(supp_large);supp_large_nan(supp_large_nan==0)=nan;

% --- Row 1: amplitude ---
clims_amp = [min(abs(E_rec_aligned).*supp_large,[],'all'),max(abs(E_rec_aligned).*supp_large,[],'all')];
ax2(1) = subplot(2,3,1);
imagesc(abs(E_rec_aligned).*supp_large_nan,clims_amp); axis image; colorbar;
title('|E| recovered');

ax2(2) = subplot(2,3,2);
imagesc(abs(E),clims_amp); axis image; colorbar;
title('|E| true');

ax2(3) = subplot(2,3,3);
imagesc(dAbsE.*supp_large_nan); axis image; colorbar;
title('\Delta |E|');

% --- Row 2: phase ---
ax2(4) = subplot(2,3,4);
imagesc(angle(E_rec_aligned).*supp_large_nan); axis image; colorbar;
title('\phi_E recovered');

ax2(5) = subplot(2,3,5);
imagesc(angle(E).*supp_large_nan); axis image; colorbar;
title('\phi_E true');

ax2(6) = subplot(2,3,6);
imagesc(dPhiE.*supp_large_nan); axis image; colorbar;
title('\Delta \phi_E (wrapped)');

colormap(parula_with_nan_white);

% link axes
linkaxes(ax2, 'xy');

total_time = toc;