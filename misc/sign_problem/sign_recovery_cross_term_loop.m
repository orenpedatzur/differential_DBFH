%% sign_recovery_cross_term_loop.m
% Clean demo for recovering Im(z) from Re(z) and |z|
% using your function:
%   solve_sign_problem_with_segments(absz, rez, support_mask, ...)
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
%
% This file loops over control parameter to identify best conditions.

clear; close all; clc;

%% ------- survey parameters ------

smooth_sigma_vec = 0.1:0.1:1;
q_threshold_vec = 10.^(-8:1:-3);


%% ---------- parameters ----------
N  = 512;
R1 = 0.25;
R2 = 0.025;

x1 = 0.00; y1 = 0.00;   % large aperture center
x2 = 0.1; y2 = 0.05;   % small aperture center (fully inside)

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

%% ---------- Fourier-domain fields ----------
A1 = fft2c(E1);
A2 = fft2c(E2);
A  = A1 + A2;

I1 = abs(A1).^2;
I  = abs(A).^2;

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


smooth_sigma_vec = 0.1:0.1:1;
q_threshold_vec = 10.^(-8:1:-3);

rel_err_imz = nan(numel(smooth_sigma_vec),numel(q_threshold_vec));

for ind_sigma = 1:numel(smooth_sigma_vec)
    parfor ind_q = 1:numel(q_threshold_vec)

        this_smooth_sigma = smooth_sigma_vec(ind_sigma);
        this_q_threshold = q_threshold_vec(ind_q);

        %% ---------- solve sign problem ----------
        [out, seg_id] = solve_sign_problem_with_segments( ...
            absz, rez, support_mask, ...
            'SmoothSigma', this_smooth_sigma, ...
            'QThresholdRel', this_q_threshold, ...
            'MinSegmentPixels', 4, ...
            'Tol', 1e-10, ...
            'MaxIter', 1e3, ...
            'LambdaSeg', 1e0, ...
            'Verbose', true, ...
            'UseSingle', false,...
            'ShowPlots',true);

        z_est = out.z_sign;
        imz_est = imag(out.z_sign);
        c_est   = ifft2c(out.z_sign);

        %% ---------- diagnostics ----------
        rel_err_imz(ind_sigma,ind_q) = norm(imz_est - imz_true, 'fro') / max(norm(imz_true, 'fro'), eps);
        rel_err_z(ind_sigma,ind_q)   = norm(z_est   - z_true,  'fro') / max(norm(z_true,  'fro'), eps);

        leak_true = norm(c_true(~support_mask)) / max(norm(c_true(:)), eps);
        leak_est  = norm(c_est (~support_mask)) / max(norm(c_est(:)),  eps);

        fprintf('relative error in Im(z): %.3e\n', rel_err_imz(ind_sigma,ind_q));
        fprintf('relative error in z:     %.3e\n', rel_err_z(ind_sigma,ind_q));
        fprintf('true leakage outside support: %.3e\n', leak_true);
        fprintf('est  leakage outside support: %.3e\n', leak_est);


    end
end

%% figures

figure;
subplot(1,2,1);
imagesc(q_threshold_vec,smooth_sigma_vec,rel_err_imz);
colorbar;
xticks(q_threshold_vec);
set(gca,'XScale','log');
xlabel('q threshold');
ylabel('sigma smooth');
title('rel err imz');
subplot(1,2,2);
imagesc(q_threshold_vec,smooth_sigma_vec,rel_err_z);
colorbar;
xticks(q_threshold_vec);
set(gca,'XScale','log');
xlabel('q threshold');
ylabel('sigma smooth');
title('rel err z');


if 0
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
    imagesc(out.q); axis image off; colorbar;
    title('q=|Im(z)|');

    axf(4) = subplot(3,4,8);
    imagesc(imfuse(seg_id==0,out.q)); axis image off; colorbar;
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
    title('|ifft2(z true)|');

    ax2 = subplot(1,3,2);
    imagesc(abs(c_est)); axis image off; colorbar;
    title('|ifft2(z est)|');

    subplot(1,3,3);
    imagesc(abs(c_est - c_true)); axis image off; colorbar;
    title('|difference|');

    linkaxes([ax1,ax2]);

end