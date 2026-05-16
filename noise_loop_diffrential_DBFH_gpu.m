%% noise_loop_differential_DBFH_gpu.m
% Full pipe-line differential-DBFH looping over noise and perturbation
% area. Boosted by GPU


clear; close all; clc;

addpath('ImZ_from_ReZ\');
addpath('diff_DBFH\');
addpath('fftc\');

rng(1);

winsomnia(true);

noise_lvl_vec = 10.^[-8:0.5:-2.5];
height_px_vec = 1:1:16;

%% ========================================================================
%  USER PARAMETERS
%  ========================================================================

N                 = 1024/1;          % grid size
seg_radius_pix    = 40/1;            % hex circumradius in pixels
num_rings         = 3;               % segmented aperture rings
gap_pix           = 1;               % physical gap between E1 and E2 supports

d_theta           = 0.1;
theta_list        = 0:d_theta:180-d_theta;

use_exact_ReZ     = false;           % false: ReZ = DeltaI/2, true: ReZ = real(Z_true)
use_true_Z_solver = false;            % true: validate DBFH using exact Z_true
% false: solve using Z_rec = ReZ_meas + 1i*ImZ_rec
show_plots        = false;
run_adjoint_tests = false;

% LSQR / preconditioner controls
nsamp             = 30;              % Hutchinson samples
blockSize         = 1;
useGPU            = true;
precond_cap       = 1e2;             % cap p <= precond_cap * median(p)
lsqr_tol          = 1e-8;
lsqr_maxit        = 1e6;

% System weights. For solve-for-E2, keep K broad; do not truncate the focal grid
% unless you are deliberately testing masking. crossWeight=100 was the best
% value from the latest sweep.
K_threshold       = 1e-3;
supportScale      = 1;
crossWeight       = 100;
% Hard anchor is enforced by removing the anchor pixel from the unknown vector.
% No soft anchor weight is used.
for ind_noise = 1:numel(noise_lvl_vec)
    for ind_px = 1:numel(height_px_vec)

        string_depth_pix  = height_px_vec(ind_px);
        noise_level       = noise_lvl_vec(ind_noise);

        %% ========================================================================
        %  BUILD SEGMENTED APERTURE AND DIFFERENTIAL SUPPORTS
        %  ========================================================================

        [seg_id, seg_centers, X, Y] = make_hex_segment_aperture(N, seg_radius_pix, num_rings);
        aperture = seg_id > 0;
        nSeg = max(seg_id(:));

        % Choose the upper-most peripheral segment as the locally perturbed region.
        [~, top_seg] = max(seg_centers(:,2));
        top_mask = seg_id == top_seg;

        % Define an outward normal through the selected segment center.
        c_top = seg_centers(top_seg,:);
        n_out = c_top / norm(c_top);

        % Signed coordinate along the outward direction.
        S = X*n_out(1) + Y*n_out(2);
        S_top = S(top_mask);
        s_edge = max(S_top);

        % E2 is a small outer sub-aperture. E1 is the rest of the aperture excluding a gap.
        cut_E2  = s_edge - string_depth_pix;
        cut_gap = cut_E2 - gap_pix;

        supp2    = top_mask & (S >= cut_E2);

        gap_mask = top_mask & (S >= cut_gap) & (S < cut_E2);
        supp1    = aperture & ~supp2 & ~gap_mask;
        supp12   = supp1 | supp2;

        %% ========================================================================
        %  RANDOM SEGMENTED PHASE FRONT
        %  ========================================================================

        E = zeros(N,N);

        for k = 1:nSeg
            mk = seg_id == k;
            if ~any(mk(:)), continue; end

            cx = seg_centers(k,1);
            cy = seg_centers(k,2);

            dx = (X - cx) / seg_radius_pix;
            dy = (Y - cy) / seg_radius_pix;

            piston = 1.0 * randn();
            tip    = 1.0 * randn();
            tilt   = 1.0 * randn();
            defoc  = 1.0 * randn();

            rho2 = dx.^2 + dy.^2;
            rho2_mean = mean(rho2(mk));

            phi = piston + tip*dx + tilt*dy + defoc*(rho2 - rho2_mean);
            E(mk) = exp(1i * phi(mk));
        end

        E1 = E .* supp1;
        E2 = E .* supp2;

        %% ========================================================================
        %  FOCAL-PLANE FIELDS AND DIFFERENTIAL CROSS TERM
        %  ========================================================================

        A1   = fft2c(E1);
        A2   = fft2c(E2);
        Atot = A1 + A2;

        I_12 = abs(Atot).^2;
        I_1  = abs(A1).^2;
        mag1 = sqrt(max(I_1, 0));        % REQUIRED input to DBFH: |A1|, not I1

        Z_true   = conj(A1) .* A2;       % the solver expects A1^* A2
        ReZ_exact = real(Z_true);
        ImZ_true  = imag(Z_true);

        I_12_noisy = I_12 + noise_level*max(I_12(:))*randn(size(I_12));
        I_1_noisy  = I_1  + noise_level*max(I_1(:))*randn(size(I_1));

        DeltaI_noisy = I_12_noisy - I_1_noisy;

        if use_exact_ReZ
            ReZ_meas = ReZ_exact;
        else
            % First-order differential approximation: DeltaI = 2 Re(A1^*A2) + |A2|^2.
            % This deliberately neglects |A2|^2.
            ReZ_meas = 0.5 * DeltaI_noisy;
        end

        % Support of g = IFT[Z]. For Z = A1^* A2 this is a correlation support.
        CSg = correlation_support(supp1, supp2);

        %% ========================================================================
        %  KK RECOVERY OF Im(Z) FROM Re(Z)
        %  ========================================================================

        [ImZ_rec, kk] = recover_ImZ_by_cartesian_KK( ...
            ReZ_meas, CSg, theta_list, ...
            'InterpMethod', 'cubic', ...
            'trueImZ', ImZ_true);

        Z_rec = ReZ_meas + 1i*ImZ_rec;

        %%
        % Cross-term diagnostic metrics.
        rel_ReZ_err = relnorm(ReZ_meas - ReZ_exact, ReZ_exact, true(size(ReZ_exact)));
        rel_ImZ_err = relnorm(ImZ_rec  - ImZ_true,  ImZ_true,  true(size(ImZ_true)));
        rel_Z_err   = relnorm(Z_rec    - Z_true,    Z_true,    true(size(Z_true)));

        fprintf('\n===== Cross-term diagnostics =====\n');
        fprintf('use_exact_ReZ                 = %d\n', use_exact_ReZ);
        fprintf('relative ReZ error            = %.3e\n', rel_ReZ_err);
        fprintf('relative ImZ error            = %.3e\n', rel_ImZ_err);
        fprintf('relative complex Z error      = %.3e\n', rel_Z_err);
        if isfield(kk, 'line_rel_err')
            fprintf('KK line-domain Im-channel err = %.3e\n', kk.line_rel_err);
        end

        %% ========================================================================
        %  EXACT FEASIBILITY CHECK BEFORE LSQR
        %  ========================================================================

        % This check bypasses the DBFH system object. If this fails, the algebra,
        % support masks, or FFT convention are wrong.
        X1_true = zeros(N,N);
        valid_A1 = mag1 > 0;
        X1_true(valid_A1) = A1(valid_A1) ./ mag1(valid_A1);

        A2_from_equation = (Z_true ./ max(mag1, eps)) .* X1_true;
        E1_check = ifft2c(mag1 .* X1_true);
        E2_check = ifft2c(A2_from_equation);

        leak1_full = support_leak(E1_check, supp1);
        leak2_full = support_leak(E2_check, supp2);

        fprintf('\n===== Full-grid true-equation check =====\n');
        fprintf('FULL true equation leaks = %.3e / %.3e\n', leak1_full, leak2_full);

        %% ========================================================================
        %  BUILD DIFFERENTIAL DBFH SYSTEM THAT SOLVES FOR E2
        %  ========================================================================

        if use_true_Z_solver
            Z_solver = Z_true;
            solver_label = 'true Z = A1^*A2';
        else
            Z_solver = Z_rec;
            solver_label = 'recovered Z = ReZ_meas + i ImZ_rec';
        end

        % Move only the DBFH/LSQR part to GPU. Keep geometry and KK recovery on CPU,
        % because interpolation/griddata is not the bottleneck and is less GPU-friendly.
        if useGPU
            assert(gpuDeviceCount > 0, 'useGPU=true, but no CUDA GPU is available.');
            gd = gpuDevice;
            fprintf('\n===== GPU device =====\n');
            fprintf('Using GPU: %s, available memory %.2f GB\n', ...
                gd.Name, gd.AvailableMemory/1024^3);

            mag1_sys     = gpuArray(mag1);
            Z_solver_sys = gpuArray(Z_solver);
            supp1_sys    = gpuArray(supp1);
            supp2_sys    = gpuArray(supp2);
        else
            mag1_sys     = mag1;
            Z_solver_sys = Z_solver;
            supp1_sys    = supp1;
            supp2_sys    = supp2;
        end

        % Unknown vector is x = [X1_on_K(except anchor) ; E2_on_supp2].
        % X1(anchor) is inserted exactly by the hard-anchor system.
        % No Z./|A1| division is used.
        if useGPU
            sys = diff_dbfh_prepare_system_gpu( ...
                mag1_sys, Z_solver_sys, supp1_sys, supp2_sys, ...
                'lambda', 0, ...
                'lambdaX1', 0, ...
                'lambdaE2', 0, ...
                'mag1RelThresh', K_threshold, ...
                'cRelThresh', 0, ...
                'supportScale', supportScale, ...
                'crossWeight', crossWeight, ...
                'anchorVal', 1, ...
                'anchorStrategy', 'maxMag1');
        else
            sys = diff_dbfh_prepare_system( ...
                mag1, Z_solver, supp1, supp2, ...
                'lambda', 0, ...
                'lambdaX1', 0, ...
                'lambdaE2', 0, ...
                'mag1RelThresh', K_threshold, ...
                'cRelThresh', 0, ...
                'supportScale', supportScale, ...
                'crossWeight', crossWeight, ...
                'anchorVal', 1, ...
                'anchorStrategy', 'maxMag1');
        end

        % Align the true fields to the solver anchor, then test the true vector.
        X1_true_aligned = X1_true;
        E2_true_aligned = E2;
        if isfield(sys, 'anchor_idx') && sys.anchor_idx > 0
            anchor_phase = X1_true(sys.anchor_idx);
            X1_true_aligned = X1_true / anchor_phase;
            E2_true_aligned = E2 / anchor_phase;
        end

        if useGPU
            X1_true_sys = gpuArray(X1_true_aligned);
            E2_true_sys = gpuArray(E2_true_aligned);
        else
            X1_true_sys = X1_true_aligned;
            E2_true_sys = E2_true_aligned;
        end

        xtrue = [X1_true_aligned(sys.Kfree); E2_true_aligned(sys.supp2)];
        rtrue = sys.Amul(xtrue) - sys.b;
        true_sys_relres = norm(rtrue) / max(norm(sys.b), eps);

        if useGPU
            out_true = diff_dbfh_unpack_solution_gpu(sys, xtrue, 'normalizeX1', false);
            out_true = gather_struct(out_true);
        else
            out_true = diff_dbfh_unpack_solution(sys, xtrue, 'normalizeX1', false);
        end

        fprintf('\n===== System feasibility check =====\n');
        fprintf('solver cross term             = %s\n', solver_label);
        fprintf('sys.n = nX1_free + nE2        = %d = %d + %d\n', sys.n, sys.nX1, sys.nE2);
        fprintf('valid X1 pixels in K          = %d / %d = %.3f\n', sys.nK, numel(sys.K), sys.nK/numel(sys.K));
        fprintf('free X1 unknowns              = %d\n', sys.nX1);
        fprintf('hard anchor index             = %d\n', sys.anchor_idx);
        fprintf('K threshold / crossWeight     = %.1e / %.3g\n', K_threshold, crossWeight);
        fprintf('TRUE [X1;E2] through sys relres = %.3e\n', true_sys_relres);
        fprintf('TRUE leak1 / cross relres      = %.3e / %.3e\n', out_true.leak1, out_true.cross_relres);

        %% ========================================================================
        %  PRECONDITIONED REAL-STACKED LSQR
        %  ========================================================================

        fprintf('\n===== Hutchinson right preconditioner =====\n');
        d = hutchinson_diag_AtA(sys, nsamp, blockSize, useGPU);
        d = double(gather(d(:)));

        dpos = d(d > 0 & isfinite(d));
        assert(~isempty(dpos), 'Hutchinson diagonal has no positive finite entries.');

        dfloor = 1e-8 * median(dpos);
        p0 = 1 ./ sqrt(max(d, dfloor));

        p0pos = gather(p0(isfinite(p0) & p0 > 0));
        p_med = median(p0pos);
        p_cap = precond_cap * p_med;
        p = min(p0, p_cap);

        fprintf('p length/sys.n = %d / %d\n', numel(p), sys.n);
        fprintf('p uncapped range = %.3e / %.3e\n', min(p0), max(p0));
        fprintf('p capped range   = %.3e / %.3e\n', min(p), max(p));
        fprintf('cap factor       = %.3e\n', precond_cap);

        Afun = make_Afun_lsqr_stacked_diagprecond(sys, p);
        bstack = [real(sys.b); imag(sys.b)];

        if run_adjoint_tests
            [stack_adj_err, true_stack_relres] = test_stacked_operator(Afun, sys, xtrue, p, bstack);
            fprintf('\n===== Real-stacked operator checks =====\n');
            fprintf('TRUE through STACKED Afun relres = %.3e\n', true_stack_relres);
            fprintf('STACKED adjoint rel err          = %.3e\n', stack_adj_err);
        end

        fprintf('\n===== LSQR solve =====\n');
        tic;
        [ustack, flag, relres, iter, resvec] = lsqr(Afun, bstack, lsqr_tol, lsqr_maxit);
        solve_time = toc

        % IMPORTANT: LSQR solved for u in A*D*u = b. Convert back to physical x.
        ur = ustack(1:sys.n);
        ui = ustack(sys.n+1:end);
        xvec = p .* (ur + 1i*ui);

        x1vec = xvec(1:sys.nX1);
        e2vec = xvec(sys.nX1+1:end);

        x1abs_cpu = gather(abs(x1vec));
        e2abs_cpu = gather(abs(e2vec));

        fprintf('flag/relres/iter             = %d / %.3e / %d\n', flag, gather(relres), iter);
        fprintf('solve time                   = %.3f s\n', solve_time);
        fprintf('|X1 free| median/min/max     = %.3e / %.3e / %.3e\n', ...
            median(x1abs_cpu), min(x1abs_cpu), max(x1abs_cpu));
        fprintf('|X1 free| p01/p99            = %.3e / %.3e\n', ...
            prctile(x1abs_cpu, 1), prctile(x1abs_cpu, 99));
        fprintf('|E2| median/min/max          = %.3e / %.3e / %.3e\n', ...
            median(e2abs_cpu), min(e2abs_cpu), max(e2abs_cpu));

        %% ========================================================================
        %  UNPACK AND ALIGN RECONSTRUCTION
        %  ========================================================================

        out_raw = diff_dbfh_unpack_solution_gpu(sys, xvec, 'normalizeX1', false);
        out_norm = diff_dbfh_unpack_solution_gpu(sys, xvec, 'normalizeX1', true);   % diagnostic only

        % Gather reconstruction products back to CPU for metrics and plotting.
        out_raw = gather_struct(out_raw);
        out_norm = gather_struct(out_norm);
        xvec_cpu = gather(xvec);
        p_cpu = gather(p);
        d_cpu = gather(d);
        resvec_cpu = gather(resvec);
        sys_plot = gather_sys_for_plot(sys);

        % Use raw. Normalizing X1 after the solve breaks the solved E2 consistency.
        out = out_raw;

        X1_rec = out.X1;
        A1_rec = out.A1;
        A2_rec = out.A2;
        E1_rec = out.E1;
        E2_rec = out.E2;
        E_rec  = out.E;

        % Remove only the global phase ambiguity using the full known pupil support.
        mask_eval = supp12 & abs(E) > 0 & abs(E_rec) > 0;
        phi0 = angle(sum(conj(E(mask_eval)) .* E_rec(mask_eval)));

        E_rec_aligned  = E_rec  .* exp(-1i*phi0);
        E1_rec_aligned = E1_rec .* exp(-1i*phi0);
        E2_rec_aligned = E2_rec .* exp(-1i*phi0);

        % Optional complex scalar alignment, useful only as a diagnostic if gain errors exist.
        alpha = sum(conj(E_rec(mask_eval)) .* E(mask_eval)) / sum(abs(E_rec(mask_eval)).^2);
        E_rec_scaled = alpha * E_rec;

        %% ========================================================================
        %  NUMERIC RECONSTRUCTION METRICS
        %  ========================================================================

        stats = struct();
        stats.raw_leak1        = out_raw.leak1;
        stats.raw_cross_relres  = out_raw.cross_relres;
        stats.norm_leak1       = out_norm.leak1;
        stats.norm_cross_relres = out_norm.cross_relres;
        stats.phase_offset_rad = phi0;
        stats.alpha_abs        = abs(alpha);
        stats.alpha_phase_rad  = angle(alpha);
        stats.field_relerr_phase_only = relnorm(E_rec_aligned - E, E, supp12);
        stats.field_relerr_complex_scalar = relnorm(E_rec_scaled - E, E, supp12);
        stats.E1_relerr        = relnorm(E1_rec_aligned - E1, E1, supp1);
        stats.E2_relerr        = relnorm(E2_rec_aligned - E2, E2, supp2);
        stats.phase_rmse_rad   = phase_rmse(E, E_rec_aligned, supp12);
        stats.phase_rmse_E1    = phase_rmse(E1, E1_rec_aligned, supp1);
        stats.phase_rmse_E2    = phase_rmse(E2, E2_rec_aligned, supp2);
        stats.iter             = iter;
        stats.relres           = relres;
        stats.flag             = flag;


        fprintf('\n===== Reconstruction metrics =====\n');
        fprintf('raw leak1 / cross relres      = %.3e / %.3e\n', stats.raw_leak1, stats.raw_cross_relres);
        fprintf('norm leak1 / cross relres     = %.3e / %.3e   (diagnostic only)\n', stats.norm_leak1, stats.norm_cross_relres);
        fprintf('global phase removed          = %.6f rad\n', stats.phase_offset_rad);
        fprintf('field relerr, phase-only      = %.3e\n', stats.field_relerr_phase_only);
        fprintf('field relerr, complex scalar  = %.3e\n', stats.field_relerr_complex_scalar);
        fprintf('complex scalar |alpha|/phase  = %.6f / %.6f rad\n', stats.alpha_abs, stats.alpha_phase_rad);
        fprintf('E1/E2 relerr                  = %.3e / %.3e\n', stats.E1_relerr, stats.E2_relerr);
        fprintf('phase RMSE full/E1/E2         = %.3e / %.3e / %.3e rad\n', ...
            stats.phase_rmse_rad, stats.phase_rmse_E1, stats.phase_rmse_E2);

        %% ========================================================================
        %  DIAGNOSTIC FIGURES
        %  ========================================================================

        if show_plots
            plot_support_diagnostics(supp1, supp2, gap_mask, CSg);
            plot_cross_term_diagnostics(Z_true, ReZ_meas, ImZ_rec, Z_rec, kk);
            plot_solver_diagnostics(sys, p, d, resvec, xvec, X1_true_aligned);
            plot_complex_field_comparison(E, E_rec_aligned, supp12,sprintf('Full pupil reconstruction: %s', solver_label));
        end
        
        %% save iteration data
        data(ind_noise,ind_px).stats = stats;
        data(ind_noise,ind_px).sys = sys;
        data(ind_noise,ind_px).out = out;
    end
end

winsomnia(false);


%% figures


flag_mat = reshape(arrayfun(@(i) data(i).stats.flag,1:numel(data)),size(data));
phase_rmse_rad_mat = reshape(arrayfun(@(i) data(i).stats.phase_rmse_rad,1:numel(data)),size(data));
raw_leak1_mat = reshape(arrayfun(@(i) data(i).stats.raw_leak1,1:numel(data)),size(data));
relres_mat = reshape(arrayfun(@(i) data(i).stats.relres,1:numel(data)),size(data));
iter_mat = reshape(arrayfun(@(i) data(i).stats.iter,1:numel(data)),size(data));

fig_error_diagnostics = figure;

ax(1) = subplot(2,3,1);
h = imagesc(height_px_vec,noise_lvl_vec,phase_rmse_rad_mat); 
set(gca,'YScale','log','YMinorTick','on','Ydir','normal');
% set(h, 'AlphaData', flag_mat==0)
xlabel('perturbation height [px]');
ylabel('normalized noise lvl');
title('phase RMSE');
colorbar;
hold on;
[C,h] = contour(height_px_vec,noise_lvl_vec,phase_rmse_rad_mat,round(10.^[-3:0.1:0],3),'color','k','linewidth',2,'linestyle',':');
clabel(C,h,'labelspacing',1e3);
colormap(ax(1),parula);

ax(2) = subplot(2,3,2);
h = imagesc(height_px_vec,noise_lvl_vec,raw_leak1_mat); 
set(gca,'YScale','log','YMinorTick','on','Ydir','normal');
% set(h, 'AlphaData', flag_mat==0)
xlabel('perturbation height [px]');
ylabel('normalized noise lvl');
title('E_1 leak');
colorbar;
hold on;
[C,h] = contour(height_px_vec,noise_lvl_vec,raw_leak1_mat,round(10.^[-3:0.1:0],3),'color','k','linewidth',2,'linestyle',':');
clabel(C,h,'labelspacing',1e3);
colormap(ax(2),parula);


ax(4) = subplot(2,3,4);
imagesc(height_px_vec,noise_lvl_vec,flag_mat); 
set(gca,'YScale','log','YMinorTick','on','Ydir','normal');
xlabel('perturbation height [px]');
ylabel('normalized noise lvl');
title('lsqr flag');
colorbar;
colormap(ax(4),gray);

ax(5) = subplot(2,3,5);
imagesc(height_px_vec,noise_lvl_vec,relres_mat); 
set(gca,'YScale','log','YMinorTick','on','Ydir','normal');
xlabel('perturbation height [px]');
ylabel('normalized noise lvl');
title('rel res');
colorbar;
colormap(ax(5),parula);

ax(6) = subplot(2,3,6);
imagesc(height_px_vec,noise_lvl_vec,iter_mat); 
set(gca,'YScale','log','YMinorTick','on','Ydir','normal');
xlabel('perturbation height [px]');
ylabel('normalized noise lvl');
title('num iteration');
colorbar;
colormap(ax(6),parula);

% saveas(fig_error_diagnostics,'K:\My Drive\differential_DBFH\figures\error_diagnostics_N1024_1E6.png');
save('K:\My Drive\differential_DBFH\data\error_diagnostics_N1024_1E6.mat')

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function CSg = correlation_support(E1_mask, E2_mask)
%CORRELATION_SUPPORT Support of IFT[A1^* A2].
%   For Z = A1^* A2, the inverse transform is a correlation-like object.
%   Its support is all displacement vectors x2 - x1.

big = conv2(double(flipud(fliplr(E1_mask))), double(E2_mask), 'same');
CSg = big > 0;

end

function [seg_id, centers, X, Y] = make_hex_segment_aperture(N, a, num_rings)
%MAKE_HEX_SEGMENT_APERTURE Build a flat-to-flat segmented hex aperture.

[xg, yg] = meshgrid(1:N,1:N);
cx = round((N+1)/2);
cy = round((N+1)/2);

X = xg - cx;
Y = cy - yg;

centers = [];
for q = -num_rings:num_rings
    for r = -num_rings:num_rings
        s = -q-r;
        if max(abs([q,r,s])) <= num_rings
            x0 = 1.5 * a * q;
            y0 = sqrt(3) * a * (r + q/2);
            centers(end+1,:) = [x0,y0]; %#ok<AGROW>
        end
    end
end

[~,ord] = sortrows([sqrt(sum(centers.^2,2)), -centers(:,2), centers(:,1)]);
centers = centers(ord,:);

seg_id = zeros(N,N);

for k = 1:size(centers,1)
    cxk = centers(k,1);
    cyk = centers(k,2);

    xv = cxk + a*cosd(0:60:300);
    yv = cyk + a*sind(0:60:300);

    colv = xv + cx;
    rowv = cy - yv;

    mk = poly2mask(colv, rowv, N, N);
    seg_id(mk) = k;
end

end

function leak = support_leak(F, support_mask)
%SUPPORT_LEAK Fraction of field energy outside a support mask.
leak = norm(F(~support_mask)) / max(norm(F(:)), eps);
end

function r = relnorm(err, ref, mask)
%RELNORM Relative norm over a mask.
if nargin < 3 || isempty(mask)
    mask = true(size(ref));
end
mask = logical(mask);
r = norm(err(mask)) / max(norm(ref(mask)), eps);
end

function rmse = phase_rmse(Ftrue, Frec, mask)
%PHASE_RMSE RMS wrapped phase error over mask.
mask = logical(mask) & abs(Ftrue) > 0 & abs(Frec) > 0;
if ~any(mask(:))
    rmse = NaN;
    return;
end
phase_err = angle(conj(Ftrue(mask)) .* Frec(mask));
rmse = sqrt(mean(phase_err.^2));
end

function plot_support_diagnostics(supp1, supp2, gap_mask, CSg)
%PLOT_SUPPORT_DIAGNOSTICS Show pupil supports and correlation support.
N = size(supp1,1);
figure('Color','w','Name','Support diagnostics','Position',[100 100 1500 450]);
tiledlayout(1,4,'TileSpacing','compact','Padding','compact');

nexttile; imagesc(supp1); axis image off; colorbar; title('supp1: exposed pupil');
nexttile; imagesc(supp2); axis image off; colorbar; title('supp2: blocked cap');
nexttile; imagesc(gap_mask); axis image off; colorbar; title('intentional gap');
nexttile; imagesc(CSg); axis image off; colorbar; title('CSg = supp IFT[A1^*A2]');
hold on;plot(N/2,N/2,'ok','MarkerFaceColor','k');
colormap parula;

end

function plot_cross_term_diagnostics(Z_true, ReZ_meas, ImZ_rec, Z_rec, kk)
%PLOT_CROSS_TERM_DIAGNOSTICS Compare measured/recovered cross term to truth.

figure('Color','w','Name','Cross-term diagnostics','Position',[80 80 1650 900]);
tiledlayout(2,4,'TileSpacing','compact','Padding','compact');

ax(1) = nexttile; imagesc(real(Z_true)); axis image off; colorbar; title('Truth Re(Z)');
ax(2) = nexttile; imagesc(ReZ_meas); axis image off; colorbar; title('Measured/approx Re(Z)');
ax(3) = nexttile; imagesc(ReZ_meas-real(Z_true)); axis image off; colorbar; title('Re(Z) error');
clim_symmetric(gca, ReZ_meas-real(Z_true));
ax(4) = nexttile; imagesc(log10(abs(Z_true)+1e-12)); axis image off; colorbar; title('log10 |Z_{true}|');

ax(5) = nexttile; imagesc(imag(Z_true)); axis image off; colorbar; title('Truth Im(Z)');
ax(6) = nexttile; imagesc(ImZ_rec); axis image off; colorbar; title('Recovered Im(Z)');
ax(7) = nexttile; imagesc(ImZ_rec-imag(Z_true)); axis image off; colorbar; title('Im(Z) error');
clim_symmetric(gca, ImZ_rec-imag(Z_true));
ax(8) = nexttile; imagesc(angle(conj(Z_true).*Z_rec)); axis image off; colorbar; title('phase error: Z_{rec} vs Z_{true}');
clim([-pi pi]);
colormap parula;
linkaxes(ax);

if isfield(kk,'gI_true_all') && ~isempty(kk.gI_true_all)
    ia = round(size(kk.gI_all,2)/4);
    figure('Color','w','Name','KK line-domain diagnostic','Position',[150 150 1200 700]);
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    nexttile; plot(real(kk.gI_true_all(:,ia)),'LineWidth',1.2); hold on; plot(real(kk.gI_all(:,ia)),'--','LineWidth',1.2); grid on; title('Re line: truth vs rec'); legend('truth','rec');xlabel('px');axis tight;
    nexttile; plot(imag(kk.gI_true_all(:,ia)),'LineWidth',1.2); hold on; plot(imag(kk.gI_all(:,ia)),'--','LineWidth',1.2); grid on; title('Im line: truth vs rec'); legend('truth','rec');xlabel('px');axis tight;
    nexttile; imagesc(kk.theta,1:size(kk.R_Re,1),real(kk.gI_all - kk.gI_true_all)); axis tight; colorbar; title('Re line-domain error');xlabel('\theta [degs]');ylabel('px');
    nexttile; imagesc(kk.theta,1:size(kk.R_Re,1),imag(kk.gI_all - kk.gI_true_all)); axis tight; colorbar; title('Im line-domain error');xlabel('\theta [degs]');ylabel('px');
    colormap parula;
end

end

function plot_solver_diagnostics(sys, p, d, resvec, xvec, X1_true_aligned)
%PLOT_SOLVER_DIAGNOSTICS Inspect preconditioner, residual convergence, and X1.
% Hard-anchor version: xvec contains X1 only on Kfree; the anchor is inserted
% explicitly for plotting.

x1vec = xvec(1:sys.nX1);
e2vec = xvec(sys.nX1+1:end);
pX = p(1:sys.nX1);
dX = d(1:sys.nX1);

P = nan(size(sys.K));
P(sys.Kfree) = pX;
P(sys.anchor_idx) = NaN;   % anchor is fixed, not preconditioned

D = nan(size(sys.K));
D(sys.Kfree) = dX;
D(sys.anchor_idx) = NaN;

Xfull = nan(size(sys.K));
Xfull(sys.Kfree) = abs(x1vec);
Xfull(sys.anchor_idx) = abs(sys.anchorVal);

phase_err = nan(size(sys.K));
if ~isempty(X1_true_aligned)
    Xrec_full = nan(size(sys.K));
    Xrec_full(sys.Kfree) = x1vec;
    Xrec_full(sys.anchor_idx) = sys.anchorVal;
    phase_err(sys.K) = angle(conj(X1_true_aligned(sys.K)) .* Xrec_full(sys.K));
end

figure('Color','w','Name','Solver diagnostics','Position',[100 100 1600 850]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile; semilogy(resvec,'LineWidth',1.2); grid on; xlabel('iteration'); ylabel('residual'); title('LSQR residual history');
nexttile; imagesc(log10(D)); axis image off; colorbar; title('log10 diag(A^*A), X1 free block');
nexttile; imagesc(log10(P)); axis image off; colorbar; title('log10 preconditioner, X1 free block');
nexttile; imagesc(Xfull); axis image off; colorbar; title('|X1 recovered| on K; anchor fixed');
nexttile; imagesc(phase_err); axis image off; colorbar; clim([-pi pi]); title('phase error of X1 on K');
nexttile; histogram(abs(e2vec),100); grid on; xlabel('|E2 unknown|'); ylabel('count'); title('|E2| histogram');
colormap parula;

end

function plot_complex_field_comparison(Ftrue, Frec, mask, figTitle)
%PLOT_COMPLEX_FIELD_COMPARISON Compare amplitude, phase, and errors.

mask = logical(mask);
amp_true = abs(Ftrue);
amp_rec  = abs(Frec);
amp_err  = amp_rec - amp_true;
ph_true  = angle(Ftrue);
ph_rec   = angle(Frec);
ph_err   = angle(conj(Ftrue).*Frec);

figure('Color','w','Name',figTitle,'Position',[70 70 1650 850]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

ax(1) = nexttile; imagesc_masked(amp_true, mask); title('|truth|');
ax(2) = nexttile; imagesc_masked(amp_rec, mask);  title('|reconstruction|');
ax(3) = nexttile; imagesc_masked(amp_err, mask);  title('amplitude error'); clim_symmetric(gca, amp_err(mask));

phase_clim = [min([ph_true(mask);ph_rec(mask)]),max([ph_true(mask);ph_rec(mask)])];
ax(4) = nexttile; imagesc_masked(ph_true, mask); title('phase truth'); clim(phase_clim);
ax(5) = nexttile; imagesc_masked(ph_rec, mask);  title('phase reconstruction, aligned'); clim(phase_clim);
ax(6) = nexttile; imagesc_masked(ph_err, mask);  title('wrapped phase error'); clim(phase_clim);
linkaxes(ax);
sgtitle(figTitle, 'Interpreter','none');
colormap parula;

end

function imagesc_masked(M, mask)
%IMAGESC_MASKED Show matrix only inside mask; outside is white.
D = M;
D(~mask) = NaN;
imagesc(D, 'AlphaData', ~isnan(D));
set(gca,'Color',[1 1 1]);
axis image off;
colorbar;
end

function clim_symmetric(ax, vals)
%CLIM_SYMMETRIC Symmetric color limits around zero.
vals = vals(:);
vals = vals(isfinite(vals));
if isempty(vals)
    a = 1;
else
    a = max(abs(vals));
    if a == 0, a = 1; end
end
clim(ax, [-a a]);
end

function S = gather_struct(S)
%GATHER_STRUCT Gather gpuArray fields in a struct. Function handles are left untouched.
fields = fieldnames(S);
for ii = 1:numel(fields)
    f = fields{ii};
    v = S.(f);
    if isa(v, 'gpuArray')
        S.(f) = gather(v);
    elseif isstruct(v)
        S.(f) = gather_struct(v);
    end
end
end

function S = gather_sys_for_plot(S)
%GATHER_SYS_FOR_PLOT Gather array fields needed by plotting/printing. Function handles remain valid but unused.
fields = fieldnames(S);
for ii = 1:numel(fields)
    f = fields{ii};
    v = S.(f);
    if isa(v, 'gpuArray')
        S.(f) = gather(v);
    end
end
end
