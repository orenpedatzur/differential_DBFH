% test for diff_DBFH in a loop over lsqr parameters
% E_true is generated in the pupil and supported on supp1.

close all; clc; clear all;

winsomnia(true);

lambda_ref_vec = 10.^(-5:1:3);
lambda_cs_vec = 10.^(-5:1:4);

%% ---------- parameters ----------
N  = 2^10;
R1 = 0.25;
R2 = 0.025;

x1 = 0.00; y1 = 0.00;   % large aperture center
d12 = R1-1.1*R2;
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

if any(supp_small(:) & ~supp_large(:))
    error('Small aperture must be fully inside the large aperture.');
end

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

figure;
subplot(1,2,1);imagesc(imfuse(real(E),supp_large-supp_small));axis equal;title('Re(E)');
subplot(1,2,2);imagesc(imfuse(imag(E),supp_large-supp_small));axis equal;title('Re(E)');

E_true = E1+E2;

% ----------------------------
% focal-plane field and intensity
% ----------------------------
A_true = fft2c(E_true);
I_full = abs(A_true).^2;

% focal-plane phasor truth
X_true = A_true ./ max(abs(A_true), eps);

% ----------------------------
% differential field generated from pupil mask
% ----------------------------
deltaE_true = double(supp2) .* E_true;
deltaA_true = fft2c(deltaE_true);

% exact cross-term
C = conj(A_true) .* deltaA_true;

%% apply noise on cross-term

C_noisy = C + 0*1e-3*max(abs(C(:)))*(randn(size(C))+1i*randn(size(C)));

%%

% ----------------------------
% run diff_DBFH
% ----------------------------
anchor_idx = sub2ind([N, N], round(N/2), round(N/2));
rel_err_X = nan(numel(lambda_cs_vec),numel(lambda_ref_vec));
rmse_phase = nan(numel(lambda_cs_vec),numel(lambda_ref_vec));

for ind_cs = 1:numel(lambda_cs_vec)
    parfor ind_ref = 1:numel(lambda_ref_vec)
    tic;
        this_lambda_cs = lambda_cs_vec(ind_cs);
        this_lambda_ref = lambda_ref_vec(ind_ref);

        out = diff_DBFH( ...
            I_full, C_noisy, supp_large, supp_small, ...
            'KRelThresh',1e-4,...
            'LambdaCS', this_lambda_cs, ...
            'LambdaRef', this_lambda_ref, ...
            'AnchorIndex', anchor_idx, ...
            'LSQRTol', 1e-5, ...
            'LSQRMaxIter', 1e3,...
            'Verbose', true,...
            'ShowPlots',false);
    toc;
        %%
        X_rec = out.X_norm;

        % ----------------------------
        % align global phase
        % ----------------------------
        alpha = sum(conj(X_rec(:)) .* X_true(:));
        alpha = alpha / max(abs(alpha), eps);
        X_rec_aligned = X_rec * alpha;

        % ----------------------------
        % errors
        % ----------------------------
        rel_err_X(ind_cs,ind_ref) = norm(X_rec_aligned(:) - X_true(:)) / max(norm(X_true(:)), eps);

        phase_err = angle(exp(1i * (angle(X_rec_aligned) - angle(X_true))));

        w = I_full./sum(I_full,'all');

        rmse_phase(ind_cs,ind_ref) = sqrt(sum(w.*phase_err.^2,'all'));

        fprintf('\nAfter global phase alignment:\n');
        fprintf('  rel error X   = %.3e\n', rel_err_X(ind_cs,ind_ref));
        fprintf('  weighted phase RMSE    = %.3e rad\n', rmse_phase(ind_cs,ind_ref));


    end
end

save('data\diff_DBFH_loop.mat');

%% figures

figure;
subplot(1,2,1);
imagesc(lambda_ref_vec,lambda_cs_vec,rel_err_X);
colorbar;
xticks(lambda_ref_vec);
yticks(lambda_cs_vec);
set(gca,'XScale','log','YScale','log');
xlabel('lambda ref');
ylabel('lambda cs smooth');
title('rel err X');
subplot(1,2,2);
imagesc(lambda_ref_vec,lambda_cs_vec,rmse_phase);
colorbar;
xticks(lambda_ref_vec);
yticks(lambda_cs_vec);
set(gca,'XScale','log','YScale','log');
xlabel('lambda ref');
ylabel('lambda cs smooth');
title('rmse phase');


winsomnia(false);

%%
% ----------------------------
% plots: pupil fields
% ----------------------------
% figure;
% 
% subplot(3,3,1);
% imagesc(abs(E_true)); axis image; colorbar;
% title('|E true|');
% 
% subplot(3,3,2);
% imagesc(angle(E_true)); axis image; colorbar;
% title('angle(E true)');
% 
% subplot(3,3,3);
% imagesc(supp_large); axis image; colorbar;
% title('CS E');
% 
% subplot(3,3,4);
% imagesc(abs(deltaE_true)); axis image; colorbar;
% title('|deltaE true|');
% 
% subplot(3,3,5);
% imagesc(angle(deltaE_true)); axis image; colorbar;
% title('angle(deltaE true)');
% 
% subplot(3,3,6);
% imagesc(supp_large); axis image; colorbar;
% title('CS E');
% 
% subplot(3,3,7);
% imagesc(log10(I_full + 1e-12)); axis image; colorbar;
% title('log10(I full)');
% 
% subplot(3,3,8);
% imagesc(real(C)); axis image; colorbar;
% title('Re(C)');
% 
% subplot(3,3,9);
% imagesc(imag(C)); axis image; colorbar;
% title('Im(C)');
% 
% colormap parula;
% % ----------------------------
% % plots: recovered focal-plane phasor + differences
% % ----------------------------
% figure();
% 
% % compute differences
% dRe   = real(X_rec_aligned) - real(X_true);
% dIm   = imag(X_rec_aligned) - imag(X_true);
% dPhi  = angle(exp(1i*(angle(X_rec_aligned) - angle(X_true)))); % wrapped
% 
% ax = gobjects(9,1);
% 
% % --- Row 1: Real ---
% ax(1) = subplot(3,3,1);
% imagesc(real(X_rec_aligned)); axis image; colorbar;
% title('Re(X) recovered');
% 
% ax(2) = subplot(3,3,2);
% imagesc(real(X_true)); axis image; colorbar;
% title('Re(X) true');
% 
% ax(3) = subplot(3,3,3);
% imagesc(dRe); axis image; colorbar;
% title('\Delta Re');
% 
% % --- Row 2: Imag ---
% ax(4) = subplot(3,3,4);
% imagesc(imag(X_rec_aligned)); axis image; colorbar;
% title('Im(X) recovered');
% 
% ax(5) = subplot(3,3,5);
% imagesc(imag(X_true)); axis image; colorbar;
% title('Im(X) true');
% 
% ax(6) = subplot(3,3,6);
% imagesc(dIm); axis image; colorbar;
% title('\Delta Im');
% 
% % --- Row 3: Phase ---
% ax(7) = subplot(3,3,7);
% imagesc(angle(X_rec_aligned)); axis image; colorbar;
% title('\phi recovered');
% 
% ax(8) = subplot(3,3,8);
% imagesc(angle(X_true)); axis image; colorbar;
% title('\phi true');
% 
% ax(9) = subplot(3,3,9);
% imagesc(dPhi); axis image; colorbar;
% title('\Delta \phi (wrapped)');
% 
% colormap parula;
% 
% % link all axes
% linkaxes(ax, 'xy');
% 
% 
% % ----------------------------
% % plots: pupil plane comparison |E| and phase
% % ----------------------------
% %%
% E_rec = out.E;
% % E_rec = ifft2c(sqrt(I_full).*(real(X_rec_aligned)+1i*imag(X_true)));
% % E_rec = ifft2c(sqrt(I_full).*(real(X_true)+1i*imag(X_rec_aligned)));
% 
% 
% % align global phase in pupil (same alpha as focal plane is fine)
% alpha_E = exp(1i * angle(sum(conj(E_rec(supp_large)) .* E_true(supp_large))));
% E_rec_aligned = E_rec * alpha_E;
% 
% % differences
% dAbsE = abs(E_rec_aligned) - abs(E_true);
% dPhiE = angle(exp(1i*(angle(E_rec_aligned) - angle(E_true))));
% 
% figure;
% 
% % --- Row 1: amplitude ---
% ax2(1) = subplot(2,3,1);
% imagesc(abs(E_rec_aligned).*supp_large); axis image; colorbar;
% title('|E| recovered');
% 
% ax2(2) = subplot(2,3,2);
% imagesc(abs(E_true)); axis image; colorbar;
% title('|E| true');
% 
% ax2(3) = subplot(2,3,3);
% imagesc(dAbsE.*supp_large); axis image; colorbar;
% title('\Delta |E|');
% 
% % --- Row 2: phase ---
% ax2(4) = subplot(2,3,4);
% imagesc(angle(E_rec_aligned).*supp_large); axis image; colorbar;
% title('\phi_E recovered');
% 
% ax2(5) = subplot(2,3,5);
% imagesc(angle(E_true)); axis image; colorbar;
% title('\phi_E true');
% 
% ax2(6) = subplot(2,3,6);
% imagesc(dPhiE.*supp_large); axis image; colorbar;
% title('\Delta \phi_E (wrapped)');
% 
% colormap parula;
% 
% % link axes
% linkaxes(ax2, 'xy');
% 

