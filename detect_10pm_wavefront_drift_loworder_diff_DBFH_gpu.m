%% detect_10pm_wavefront_drift_loworder_diff_DBFH_gpu_corefigs_fixedSNR.m
% Numerical detectability study for <=10 pm segmented low-order wavefront drift using diff-DBFH.
%
% Question answered by this script:
%   For what focal-plane SNR / additive noise level can diff-DBFH detect a
%   small wavefront change, here a <=10 pm random segmented low-order drift?
%
% Important: this script does NOT compare to dOTF. The observable is the
% reconstructed DIFFERENCE between two diff-DBFH reconstructions. This is the
% relevant metric because diff-DBFH may have a stable reconstruction bias; a
% small wavefront change can still be detectable if that bias is common-mode.
%
% Requires the same helper folders used by your existing numerical script:
%   ImZ_from_ReZ\, diff_DBFH\, fftc\

clear; clc;close all;

addpath('ImZ_from_ReZ\');
addpath('diff_DBFH\');
addpath('fftc\');

rng(1);

winsomnia(true);


%% ========================================================================
%  USER PARAMETERS
%  ========================================================================

% Physical phase conversion.
lambda_m                     = 633e-9;
drift_target_pm              = 10;       % upper limit / target total WFE RMS in pm
surface_to_wavefront_factor  = 1;        % keep 1 when drift_target_pm is already wavefront error

drift_definition = 'rms_segment_loworder';
% Options:
%   'rms_segment_loworder'       : random segment piston/tip/tilt/defocus, full-aperture RMS = drift_target_pm
%   'bounded_segment_loworder'   : random segment piston/tip/tilt/defocus, max abs WFE = drift_target_pm
%   'rms_segment_piston'         : legacy piston-only, full-aperture RMS = drift_target_pm
%   'bounded_segment_piston'     : legacy piston-only, max abs WFE = drift_target_pm
%
% For a strict "up to 10 pm" ensemble rather than exactly 10 pm RMS, set
% drift_random_scale_range = [0 1]. For threshold testing, [1 1] is cleaner.
drift_random_scale_range = [1 1];

% Numerical aperture model.
N                 = 1024;
seg_radius_pix    = 40;       % hex circumradius in pixels, matching your numerical study
num_rings         = 3;
gap_pix           = 1;
string_depth_pix  = 17;       % fixed perturbation height. Change this to the best value from prior sweep.

% Noise sweep. noise_level multiplies max(I) before additive Gaussian noise.
% For 10 pm phase (~1e-4 rad at 633 nm), you need to see both the high-SNR
% floor and the low-SNR failure regime. This range gives an injected-SNR span
% of roughly 1e4-1e10 for the default PSF normalization.
noise_lvl_vec = 10.^(-13:1:-7);
n_trials_per_noise = 5;       % each trial runs THREE diff-DBFH solves; use 20+ for a real threshold claim

% Focal-plane SNR calibration.
% IMPORTANT:
%   In simulation, use the known injected noise realization:
%       noise = std((I_noisy - I_clean) in corner 21x21)
%   A single-image corner std of I_noisy is NOT a clean noise estimate here,
%   because the noiseless PSF has deterministic structure in the corner and
%   creates an artificial SNR floor/plateau.
snr_window_half_width = 10;    % gives 21x21 windows

% KK recovery.
d_theta    = 0.1;
theta_list = 0:d_theta:180-d_theta;

% Differential DBFH controls.
use_exact_ReZ     = false;     % false: ReZ = DeltaI/2 from noisy measurements
use_true_Z_solver = false;     % true only for sanity checks; not realistic
use_noisy_mag1    = true;      % realistic: |A1| comes from the noisy perturbed intensity
clip_negative_I   = true;

% LSQR / preconditioner controls.
nsamp             = 30;
blockSize         = 1;
useGPU            = true;
precond_cap       = 1e2;
lsqr_tol          = 1e-8;
lsqr_maxit        = 1e4;

% System weights.
K_threshold       = 1e-3;
supportScale      = 1;
crossWeight       = 100;

% Detection rule for the summary table. This is intentionally separated from
% the raw metrics; do not hide behind one arbitrary threshold.
detection_sigma_threshold = 3;     % matched-filter signal relative to null scatter
corr_threshold            = 0.30;  % weak but positive pattern agreement; tighten later if needed
accuracy_fraction_thresh  = 1.0;   % drift RMSE less than true drift RMS means rough recoverability

% Output.
result_dir = fullfile(pwd, ['diff_dbfh_10pm_loworder_detectability_' datestr(now,'yyyymmdd_HHMMSS')]);
if ~exist(result_dir, 'dir'), mkdir(result_dir); end
make_summary_plots = true;

params = struct();
params.lambda_m = lambda_m;
params.use_exact_ReZ = use_exact_ReZ;
params.use_true_Z_solver = use_true_Z_solver;
params.use_noisy_mag1 = use_noisy_mag1;
params.clip_negative_I = clip_negative_I;
params.theta_list = theta_list;
params.nsamp = nsamp;
params.blockSize = blockSize;
params.useGPU = useGPU;
params.precond_cap = precond_cap;
params.lsqr_tol = lsqr_tol;
params.lsqr_maxit = lsqr_maxit;
params.K_threshold = K_threshold;
params.supportScale = supportScale;
params.crossWeight = crossWeight;
params.snr_window_half_width = snr_window_half_width;
params.drift_random_scale_range = drift_random_scale_range;

%% ========================================================================
%  BUILD GEOMETRY ONCE
%  ========================================================================

fprintf('Building aperture and differential supports...\n');
[seg_id, seg_centers, X, Y] = make_hex_segment_aperture(N, seg_radius_pix, num_rings);
aperture = seg_id > 0;
nSeg = max(seg_id(:));

[~, top_seg] = max(seg_centers(:,2));
top_mask = seg_id == top_seg;
c_top = seg_centers(top_seg,:);
n_out = c_top / norm(c_top);
S = X*n_out(1) + Y*n_out(2);
s_edge = max(S(top_mask));

cut_E2  = s_edge - string_depth_pix;
cut_gap = cut_E2 - gap_pix;

supp2    = top_mask & (S >= cut_E2);
gap_mask = top_mask & (S >= cut_gap) & (S < cut_E2);
supp1    = aperture & ~supp2 & ~gap_mask;
supp12   = supp1 | supp2;
CSg      = correlation_support(supp1, supp2);

fprintf('N = %d, segments = %d, perturbation pixels = %d, support pixels = %d\n', ...
    N, nSeg, nnz(supp2), nnz(supp12));
fprintf('Drift WFE limit/target = %.3g pm, definition = %s, random scale range = [%.2g %.2g]\n', ...
    drift_target_pm, drift_definition, drift_random_scale_range(1), drift_random_scale_range(2));
fprintf('Output directory: %s\n', result_dir);

%% ========================================================================
%  MAIN SWEEP
%  ========================================================================

n_total = numel(noise_lvl_vec) * n_trials_per_noise;
records = repmat(make_record_template(), n_total, 1);
rec_idx = 0;

for ind_noise = 1:numel(noise_lvl_vec)
    noise_level = noise_lvl_vec(ind_noise);
    fprintf('\n=================================================================\n');
    fprintf('Noise %d/%d: noise_level = %.3e\n', ind_noise, numel(noise_lvl_vec), noise_level);
    fprintf('=================================================================\n');

    for trial = 1:n_trials_per_noise
        rec_idx = rec_idx + 1;
        rec = make_record_template();
        rec.noise_index = ind_noise;
        rec.trial = trial;
        rec.noise_level = noise_level;
        rec.drift_target_pm = drift_target_pm;
        rec.string_depth_pix = string_depth_pix;
        rec.N = N;
        rec.seg_radius_pix = seg_radius_pix;

        fprintf('\n--- trial %d/%d at noise %.3e ---\n', trial, n_trials_per_noise, noise_level);

        try
            % Base wavefront and a known small segmented drift.
            E0 = make_random_segmented_field(seg_id, seg_centers, X, Y, seg_radius_pix, nSeg);
            [dphi_drift, drift_pm_by_seg, drift_info] = make_segment_loworder_drift( ...
                seg_id, seg_centers, X, Y, seg_radius_pix, supp12, nSeg, lambda_m, ...
                drift_target_pm, drift_definition, surface_to_wavefront_factor, drift_random_scale_range);
            E1 = E0 .* exp(1i * dphi_drift) .* supp12;

            mask_eval = supp12 & abs(E0) > 0 & abs(E1) > 0;
            dphi_true = remove_piston(angle(conj(E0) .* E1), mask_eval);
            true_drift_rms_pm = phase_rms_to_pm(masked_rms(dphi_true, mask_eval), lambda_m);
            true_drift_pv_pm  = phase_rad_to_pm(max(dphi_true(mask_eval)) - min(dphi_true(mask_eval)), lambda_m);

            % Three reconstructions:
            %   A: base wavefront, noise realization 1
            %   B: same base wavefront, noise realization 2  -> null/no-change baseline
            %   C: drifted wavefront, noise realization 3    -> signal/change measurement
            outA = run_diff_dbfh_once(E0, supp1, supp2, CSg, noise_level, params);
            outB = run_diff_dbfh_once(E0, supp1, supp2, CSg, noise_level, params);
            outC = run_diff_dbfh_once(E1, supp1, supp2, CSg, noise_level, params);

            EA = outA.E_rec;
            EB = outB.E_rec;
            EC = outC.E_rec;

            mask_rec = mask_eval & abs(EA) > 0 & abs(EB) > 0 & abs(EC) > 0;

            % Reconstruction quality vs known truth. This is a sanity metric, not
            % the operational detection statistic.
            EA_truth_aligned = align_global_phase(E0, EA, mask_rec);
            EC_truth_aligned = align_global_phase(E1, EC, mask_rec);
            rec.phase_rmse_A_vs_truth_pm = phase_rms_to_pm(phase_rmse(E0, EA_truth_aligned, mask_rec), lambda_m);
            rec.phase_rmse_C_vs_truth_pm = phase_rms_to_pm(phase_rmse(E1, EC_truth_aligned, mask_rec), lambda_m);

            % Operational change measurements. Global piston is unobservable and
            % also absorbs the arbitrary reconstruction phase.
            dphi_null = remove_piston(angle(conj(EA) .* EB), mask_rec);
            dphi_sig  = remove_piston(angle(conj(EA) .* EC), mask_rec);

            dphi_err = remove_piston(wrap_phase(dphi_sig - dphi_true), mask_rec);

            rec.true_drift_rms_pm = true_drift_rms_pm;
            rec.true_drift_pv_pm = true_drift_pv_pm;
            rec.drift_seg_rms_pm = safe_rms(drift_pm_by_seg - mean(drift_pm_by_seg));
            rec.drift_seg_maxabs_pm = max(abs(drift_pm_by_seg));
            rec.drift_coeff_rms_pm = drift_info.coeff_rms_pm;
            rec.drift_wfe_maxabs_pm = drift_info.wfe_maxabs_pm;
            rec.drift_wfe_pv_pm = drift_info.wfe_pv_pm;
            rec.drift_scale_factor = drift_info.scale_factor;
            rec.drift_actual_target_pm = drift_info.actual_target_pm;

            rec.snr_A = outA.est_snr;
            rec.snr_B = outB.est_snr;
            rec.snr_C = outC.est_snr;
            rec.snr_mean = mean([outA.est_snr, outB.est_snr, outC.est_snr], 'omitnan');
            rec.snr_spatial_A = outA.est_snr_spatial_corner;
            rec.snr_spatial_B = outB.est_snr_spatial_corner;
            rec.snr_spatial_C = outC.est_snr_spatial_corner;
            rec.snr_spatial_mean = mean([outA.est_snr_spatial_corner, outB.est_snr_spatial_corner, outC.est_snr_spatial_corner], 'omitnan');

            rec.null_change_rms_pm   = phase_rms_to_pm(masked_rms(dphi_null, mask_rec), lambda_m);
            rec.signal_change_rms_pm = phase_rms_to_pm(masked_rms(dphi_sig,  mask_rec), lambda_m);
            rec.drift_error_rmse_pm  = phase_rms_to_pm(masked_rms(dphi_err,  mask_rec), lambda_m);

            rec.null_projection_pm   = project_onto_true_drift_pm(dphi_null, dphi_true, mask_rec, lambda_m);
            rec.signal_projection_pm = project_onto_true_drift_pm(dphi_sig,  dphi_true, mask_rec, lambda_m);
            rec.drift_corr           = safe_corr(dphi_sig(mask_rec), dphi_true(mask_rec));
            rec.null_corr            = safe_corr(dphi_null(mask_rec), dphi_true(mask_rec));

            rec.iter_A = outA.iter;
            rec.iter_B = outB.iter;
            rec.iter_C = outC.iter;
            rec.relres_A = outA.relres;
            rec.relres_B = outB.relres;
            rec.relres_C = outC.relres;
            rec.flag_A = outA.flag;
            rec.flag_B = outB.flag;
            rec.flag_C = outC.flag;
            rec.raw_leak1_A = outA.raw_leak1;
            rec.raw_leak1_C = outC.raw_leak1;
            rec.cross_relres_A = outA.raw_cross_relres;
            rec.cross_relres_C = outC.raw_cross_relres;

            rec.success = true;
            rec.error_message = "";

            fprintf('SNR mean %.3e | true drift RMS %.3f pm | null RMS %.3f pm | signal RMS %.3f pm | proj %.3f pm | err %.3f pm | corr %.3f\n', ...
                rec.snr_mean, rec.true_drift_rms_pm, rec.null_change_rms_pm, rec.signal_change_rms_pm, ...
                rec.signal_projection_pm, rec.drift_error_rmse_pm, rec.drift_corr);

        catch ME
            rec.success = false;
            rec.error_message = string(ME.message);
            fprintf(2, 'FAILED trial %d at noise %.3e: %s\n', trial, noise_level, ME.message);
        end

        records(rec_idx) = rec;

        % Live save after each trial. MATLAB crashes happen; do not lose the sweep.
        records_live = records(1:rec_idx);
        T_live = struct2table(records_live, 'AsArray', true);
        writetable(T_live, fullfile(result_dir, 'diff_dbfh_10pm_detectability_trials_live.csv'));
        save(fullfile(result_dir, 'diff_dbfh_10pm_detectability_live.mat'), ...
            'records_live', 'noise_lvl_vec', 'params', 'drift_target_pm', 'drift_definition', ...
            'string_depth_pix', 'seg_radius_pix', 'N', 'drift_random_scale_range', '-v7.3');
    end
end

records = records(1:rec_idx);
T = struct2table(records, 'AsArray', true);
writetable(T, fullfile(result_dir, 'diff_dbfh_10pm_detectability_trials.csv'));

Ssummary = summarize_detectability(T, detection_sigma_threshold, corr_threshold, accuracy_fraction_thresh);
writetable(Ssummary, fullfile(result_dir, 'diff_dbfh_10pm_detectability_summary.csv'));

save(fullfile(result_dir, 'diff_dbfh_10pm_detectability_results.mat'), ...
    'records', 'T', 'Ssummary', 'noise_lvl_vec', 'params', 'drift_target_pm', 'drift_definition', ...
    'string_depth_pix', 'seg_radius_pix', 'N', 'drift_random_scale_range', '-v7.3');

write_threshold_report(Ssummary, result_dir, detection_sigma_threshold, corr_threshold, accuracy_fraction_thresh);

if make_summary_plots
    try
        plot_detectability_summary(Ssummary, result_dir, drift_target_pm);
    catch ME
        warning('Summary plotting failed: %s', ME.message);
    end
end

fprintf('\nDone. Results saved in:\n%s\n', result_dir);

%% ========================================================================
%  CORE SOLVER WRAPPER
%  ========================================================================

function out = run_diff_dbfh_once(E, supp1, supp2, CSg, noise_level, params)
%RUN_DIFF_DBFH_ONCE Generate two noisy focal-plane inputs and solve diff-DBFH.

    E1 = E .* supp1;
    E2 = E .* supp2;

    A1   = fft2c(E1);
    A2   = fft2c(E2);
    Atot = A1 + A2;

    I_12 = abs(Atot).^2;
    I_1  = abs(A1).^2;

    sigma12 = noise_level * max(I_12(:));
    sigma1  = noise_level * max(I_1(:));

    I_12_noisy = I_12 + sigma12 * randn(size(I_12));
    I_1_noisy  = I_1  + sigma1  * randn(size(I_1));

    if params.clip_negative_I
        I_12_for_mag = max(I_12_noisy, 0);
        I_1_for_mag  = max(I_1_noisy, 0);
    else
        I_12_for_mag = I_12_noisy;
        I_1_for_mag  = I_1_noisy;
    end

    if params.use_noisy_mag1
        mag1 = sqrt(max(I_1_for_mag, 0));
    else
        mag1 = sqrt(max(I_1, 0));
    end

    Z_true = conj(A1) .* A2;
    ReZ_exact = real(Z_true);
    ImZ_true = imag(Z_true);

    if params.use_exact_ReZ
        ReZ_meas = ReZ_exact;
    else
        DeltaI_noisy = I_12_noisy - I_1_noisy;
        ReZ_meas = 0.5 * DeltaI_noisy;
    end

    [ImZ_rec, ~] = recover_ImZ_by_cartesian_KK( ...
        ReZ_meas, CSg, params.theta_list, ...
        'InterpMethod', 'cubic', ...
        'trueImZ', ImZ_true);

    if params.use_true_Z_solver
        Z_solver = Z_true;
    else
        Z_solver = ReZ_meas + 1i * ImZ_rec;
    end

    if params.useGPU
        assert(gpuDeviceCount > 0, 'useGPU=true, but no CUDA GPU is available.');
        mag1_sys     = gpuArray(mag1);
        Z_solver_sys = gpuArray(Z_solver);
        supp1_sys    = gpuArray(supp1);
        supp2_sys    = gpuArray(supp2);

        sys = diff_dbfh_prepare_system_gpu( ...
            mag1_sys, Z_solver_sys, supp1_sys, supp2_sys, ...
            'lambda', 0, ...
            'lambdaX1', 0, ...
            'lambdaE2', 0, ...
            'mag1RelThresh', params.K_threshold, ...
            'cRelThresh', 0, ...
            'supportScale', params.supportScale, ...
            'crossWeight', params.crossWeight, ...
            'anchorVal', 1, ...
            'anchorStrategy', 'maxMag1');
    else
        sys = diff_dbfh_prepare_system( ...
            mag1, Z_solver, supp1, supp2, ...
            'lambda', 0, ...
            'lambdaX1', 0, ...
            'lambdaE2', 0, ...
            'mag1RelThresh', params.K_threshold, ...
            'cRelThresh', 0, ...
            'supportScale', params.supportScale, ...
            'crossWeight', params.crossWeight, ...
            'anchorVal', 1, ...
            'anchorStrategy', 'maxMag1');
    end

    d = hutchinson_diag_AtA(sys, params.nsamp, params.blockSize, params.useGPU);
    d = d(:);
    d_cpu = to_cpu(d);

    dpos = d_cpu(d_cpu > 0 & isfinite(d_cpu));
    assert(~isempty(dpos), 'Hutchinson diagonal has no positive finite entries.');

    dfloor = 1e-8 * median(dpos);
    p0 = 1 ./ sqrt(max(d, dfloor));

    p0_cpu = to_cpu(p0);
    p0pos = p0_cpu(isfinite(p0_cpu) & p0_cpu > 0);
    p_med = median(p0pos);
    p_cap = params.precond_cap * p_med;
    p = min(p0, p_cap);

    Afun = make_Afun_lsqr_stacked_diagprecond(sys, p);
    bstack = [real(sys.b); imag(sys.b)];

    [ustack, flag, relres, iter, ~] = lsqr(Afun, bstack, params.lsqr_tol, params.lsqr_maxit);

    ur = ustack(1:sys.n);
    ui = ustack(sys.n+1:end);
    xvec = p .* (ur + 1i * ui);

    if params.useGPU
        out_raw = diff_dbfh_unpack_solution_gpu(sys, xvec, 'normalizeX1', false);
    else
        out_raw = diff_dbfh_unpack_solution(sys, xvec, 'normalizeX1', false);
    end
    out_raw = gather_struct(out_raw);

    out = struct();
    out.E_rec = out_raw.E;
    out.E1_rec = out_raw.E1;
    out.E2_rec = out_raw.E2;
    out.raw_leak1 = out_raw.leak1;
    out.raw_cross_relres = out_raw.cross_relres;
    out.flag = flag;
    out.relres = double(to_cpu(relres));
    out.iter = iter;

    % Use the injected-noise realization for the simulated SNR calibration.
    % This should scale approximately as 1/noise_level on a log-log plot.
    % The old single-image corner estimator is also kept as a diagnostic, but
    % it should not be used as the simulation control-axis SNR.
    out.est_snr = estimate_injected_focal_snr(I_12, I_12_noisy, params.snr_window_half_width);
    out.est_snr_spatial_corner = estimate_single_image_corner_snr(I_12_noisy, params.snr_window_half_width);
    out.noise_sigma12 = sigma12;
    out.noise_sigma1 = sigma1;
end

%% ========================================================================
%  SUMMARY AND PLOTTING
%  ========================================================================

function S = summarize_detectability(T, detection_sigma_threshold, corr_threshold, accuracy_fraction_thresh)
    Tgood = T(T.success == true, :);
    if isempty(Tgood)
        S = table();
        return;
    end

    noise_vals = unique(T.noise_level, 'stable');
    rows = repmat(make_summary_template(), numel(noise_vals), 1);

    for ii = 1:numel(noise_vals)
        nl = noise_vals(ii);
        tall = T(T.noise_level == nl, :);
        tg = Tgood(Tgood.noise_level == nl, :);

        rec = make_summary_template();
        rec.noise_level = nl;
        rec.n_trials_total = height(tall);
        rec.n_success = height(tg);

        if ~isempty(tg)
            rec.median_snr = median(tg.snr_mean, 'omitnan');
            rec.q25_snr = prctile(tg.snr_mean, 25);
            rec.q75_snr = prctile(tg.snr_mean, 75);
            if ismember('snr_spatial_mean', tg.Properties.VariableNames)
                rec.median_spatial_snr = median(tg.snr_spatial_mean, 'omitnan');
            end

            rec.median_true_drift_rms_pm = median(tg.true_drift_rms_pm, 'omitnan');
            rec.median_null_change_rms_pm = median(tg.null_change_rms_pm, 'omitnan');
            rec.median_signal_change_rms_pm = median(tg.signal_change_rms_pm, 'omitnan');
            rec.median_drift_error_rmse_pm = median(tg.drift_error_rmse_pm, 'omitnan');
            rec.median_signal_projection_pm = median(tg.signal_projection_pm, 'omitnan');
            rec.median_null_projection_pm = median(tg.null_projection_pm, 'omitnan');
            rec.mean_signal_projection_pm = mean(tg.signal_projection_pm, 'omitnan');
            rec.mean_null_projection_pm = mean(tg.null_projection_pm, 'omitnan');
            rec.std_null_projection_pm = std(tg.null_projection_pm, 0, 'omitnan');
            rec.std_signal_projection_pm = std(tg.signal_projection_pm, 0, 'omitnan');
            rec.median_drift_corr = median(tg.drift_corr, 'omitnan');
            rec.median_null_corr = median(tg.null_corr, 'omitnan');
            rec.median_phase_rmse_truth_pm = median(tg.phase_rmse_A_vs_truth_pm, 'omitnan');

            if rec.std_null_projection_pm > 0 && isfinite(rec.std_null_projection_pm)
                rec.detection_z = abs(rec.mean_signal_projection_pm - rec.mean_null_projection_pm) / rec.std_null_projection_pm;
            end

            rec.detectable_3sigma = rec.detection_z >= detection_sigma_threshold;
            rec.pattern_positive = rec.median_drift_corr >= corr_threshold;
            rec.roughly_recoverable = rec.median_drift_error_rmse_pm <= accuracy_fraction_thresh * rec.median_true_drift_rms_pm;
            rec.pass_combined_rule = rec.detectable_3sigma && rec.pattern_positive;
        end

        rows(ii) = rec;
    end

    S = struct2table(rows, 'AsArray', true);
    S = sortrows(S, 'median_snr', 'ascend');
end

function write_threshold_report(S, result_dir, detection_sigma_threshold, corr_threshold, accuracy_fraction_thresh)
    report_path = fullfile(result_dir, 'threshold_report.txt');
    fid = fopen(report_path, 'w');
    if fid < 0
        warning('Could not write threshold report.');
        return;
    end
    cleaner = onCleanup(@() fclose(fid));

    fprintf(fid, 'diff-DBFH <=10 pm low-order drift detectability report\n');
    fprintf(fid, '=========================================\n\n');
    fprintf(fid, 'Detection rule:\n');
    fprintf(fid, '  matched-filter detection_z >= %.3g\n', detection_sigma_threshold);
    fprintf(fid, '  median drift correlation >= %.3g\n', corr_threshold);
    fprintf(fid, 'Accuracy diagnostic, not required for detection:\n');
    fprintf(fid, '  median drift-error RMSE <= %.3g x true drift RMS\n\n', accuracy_fraction_thresh);

    if isempty(S)
        fprintf(fid, 'No successful trials.\n');
        return;
    end

    pass = S.pass_combined_rule == true & isfinite(S.median_snr);
    if any(pass)
        first = find(pass, 1, 'first');
        fprintf(fid, 'First passing SNR point:\n');
        fprintf(fid, '  median SNR              = %.6e\n', S.median_snr(first));
        fprintf(fid, '  input noise level        = %.6e\n', S.noise_level(first));
        fprintf(fid, '  detection_z              = %.6g\n', S.detection_z(first));
        fprintf(fid, '  median drift corr        = %.6g\n', S.median_drift_corr(first));
        fprintf(fid, '  signal projection        = %.6g pm\n', S.median_signal_projection_pm(first));
        fprintf(fid, '  null projection          = %.6g pm\n', S.median_null_projection_pm(first));
        fprintf(fid, '  drift-error RMSE         = %.6g pm\n', S.median_drift_error_rmse_pm(first));
        fprintf(fid, '  true drift RMS           = %.6g pm\n', S.median_true_drift_rms_pm(first));
    else
        fprintf(fid, 'No SNR point passed the combined detection rule. Expand the SNR sweep lower-noise/higher-SNR.\n');
    end
end

function plot_detectability_summary(S, result_dir, drift_target_pm)
%PLOT_DETECTABILITY_SUMMARY Minimal paper-facing figures.
%
% Keep this deliberately narrow:
%   1. SNR calibration: injected normalized noise level -> measured focal-plane SNR.
%   2. Detectability: matched-filter detection metric -> measured focal-plane SNR.
%
% Other diagnostics belong in the CSV/MAT file, not in the main figure set.

    if isempty(S), return; end

    good = isfinite(S.median_snr) & S.n_success > 0;
    S = S(good, :);
    if isempty(S), return; end

    % --------------------------------------------------------------------
    % Figure 1: SNR response to injected noise level.
    % This is the calibration curve connecting the simulation control knob
    % to the observable focal-plane SNR.
    % --------------------------------------------------------------------
    Snoise = sortrows(S, 'noise_level', 'ascend');

    fig = figure('Color','w','Visible','on','Position',[100 100 900 650]);
    loglog(Snoise.noise_level, Snoise.median_snr, 'o-', 'LineWidth', 1.8); hold on; grid on;

    % Show interquartile SNR range when available. Do not let errorbar crash
    % older MATLAB releases with table variables that contain NaNs.
    lo = Snoise.median_snr - Snoise.q25_snr;
    hi = Snoise.q75_snr - Snoise.median_snr;
    ok = isfinite(Snoise.noise_level) & isfinite(Snoise.median_snr) & isfinite(lo) & isfinite(hi) & lo >= 0 & hi >= 0;
    if any(ok)
        errorbar(Snoise.noise_level(ok), Snoise.median_snr(ok), lo(ok), hi(ok), ...
            'LineStyle','none', 'LineWidth', 1.0);
    end

    set(gca, 'XDir', 'reverse');
    xlabel('injected normalized noise level');
    ylabel('estimated focal-plane SNR');
    title('Injected-noise SNR calibration for simulated focal-plane captures');
    safe_save_png(fig, fullfile(result_dir, 'snr_vs_noise_level.png'));
    % close(fig);

    % --------------------------------------------------------------------
    % Figure 2: Detection metric versus measured SNR.
    % detection_z is the separation between the true-drift matched projection
    % and the no-change/null projection, normalized by null scatter.
    % --------------------------------------------------------------------
    Ssnr = sortrows(S, 'median_snr', 'ascend');

    fig = figure('Color','w','Visible','on','Position',[100 100 900 650]);
    semilogx(Ssnr.median_snr, Ssnr.detection_z, 'o-', 'LineWidth', 1.8); hold on; grid on;
    yline(3, ':', '3\sigma threshold', 'LineWidth', 1.2);
    xlabel('estimated focal-plane SNR');
    ylabel('detection metric, z');
    title(sprintf('Detectability of %.3g pm segmented low-order wavefront drift', drift_target_pm));
    safe_save_png(fig, fullfile(result_dir, 'detection_metric_vs_snr.png'));
    % close(fig);
end

%% ========================================================================
%  RECORD TEMPLATES
%  ========================================================================

function rec = make_record_template()
    rec = struct();
    rec.success = false;
    rec.error_message = "";
    rec.noise_index = NaN;
    rec.trial = NaN;
    rec.noise_level = NaN;
    rec.N = NaN;
    rec.seg_radius_pix = NaN;
    rec.string_depth_pix = NaN;
    rec.drift_target_pm = NaN;
    rec.true_drift_rms_pm = NaN;
    rec.true_drift_pv_pm = NaN;
    rec.drift_seg_rms_pm = NaN;
    rec.drift_seg_maxabs_pm = NaN;
    rec.drift_coeff_rms_pm = NaN;
    rec.drift_wfe_maxabs_pm = NaN;
    rec.drift_wfe_pv_pm = NaN;
    rec.drift_scale_factor = NaN;
    rec.drift_actual_target_pm = NaN;
    rec.snr_A = NaN;
    rec.snr_B = NaN;
    rec.snr_C = NaN;
    rec.snr_mean = NaN;
    rec.snr_spatial_A = NaN;
    rec.snr_spatial_B = NaN;
    rec.snr_spatial_C = NaN;
    rec.snr_spatial_mean = NaN;
    rec.null_change_rms_pm = NaN;
    rec.signal_change_rms_pm = NaN;
    rec.drift_error_rmse_pm = NaN;
    rec.null_projection_pm = NaN;
    rec.signal_projection_pm = NaN;
    rec.drift_corr = NaN;
    rec.null_corr = NaN;
    rec.phase_rmse_A_vs_truth_pm = NaN;
    rec.phase_rmse_C_vs_truth_pm = NaN;
    rec.iter_A = NaN;
    rec.iter_B = NaN;
    rec.iter_C = NaN;
    rec.relres_A = NaN;
    rec.relres_B = NaN;
    rec.relres_C = NaN;
    rec.flag_A = NaN;
    rec.flag_B = NaN;
    rec.flag_C = NaN;
    rec.raw_leak1_A = NaN;
    rec.raw_leak1_C = NaN;
    rec.cross_relres_A = NaN;
    rec.cross_relres_C = NaN;
end

function rec = make_summary_template()
    rec = struct();
    rec.noise_level = NaN;
    rec.n_trials_total = NaN;
    rec.n_success = NaN;
    rec.median_snr = NaN;
    rec.q25_snr = NaN;
    rec.q75_snr = NaN;
    rec.median_spatial_snr = NaN;
    rec.median_true_drift_rms_pm = NaN;
    rec.median_null_change_rms_pm = NaN;
    rec.median_signal_change_rms_pm = NaN;
    rec.median_drift_error_rmse_pm = NaN;
    rec.median_signal_projection_pm = NaN;
    rec.median_null_projection_pm = NaN;
    rec.mean_signal_projection_pm = NaN;
    rec.mean_null_projection_pm = NaN;
    rec.std_null_projection_pm = NaN;
    rec.std_signal_projection_pm = NaN;
    rec.detection_z = NaN;
    rec.median_drift_corr = NaN;
    rec.median_null_corr = NaN;
    rec.median_phase_rmse_truth_pm = NaN;
    rec.detectable_3sigma = false;
    rec.pattern_positive = false;
    rec.roughly_recoverable = false;
    rec.pass_combined_rule = false;
end

%% ========================================================================
%  WAVEFRONT GENERATION AND PHASE METRICS
%  ========================================================================

function E = make_random_segmented_field(seg_id, seg_centers, X, Y, seg_radius_pix, nSeg)
    E = zeros(size(seg_id));
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
end

function [dphi_map, drift_pm_by_seg, info] = make_segment_loworder_drift( ...
    seg_id, seg_centers, X, Y, seg_radius_pix, supp12, nSeg, lambda_m, ...
    drift_target_pm, drift_definition, surface_to_wavefront_factor, drift_random_scale_range)
%MAKE_SEGMENT_LOWORDER_DRIFT Random segmented WFE drift.
%
% The main mode is random piston/tip/tilt/defocus on every hex segment,
% normalized by the TOTAL pupil WFE after removing global piston.
%
% Output dphi_map is phase [rad]. The pm quantities are wavefront error, not
% mirror surface height, unless surface_to_wavefront_factor is deliberately changed.

    if nargin < 12 || isempty(drift_random_scale_range)
        drift_random_scale_range = [1 1];
    end
    if numel(drift_random_scale_range) == 1
        drift_random_scale_range = [drift_random_scale_range drift_random_scale_range];
    end
    drift_random_scale_range = sort(double(drift_random_scale_range(:).'));
    target_scale = drift_random_scale_range(1) + diff(drift_random_scale_range) * rand();
    actual_target_pm = drift_target_pm * target_scale;

    mask = logical(supp12);
    wfe_pm = zeros(size(seg_id));
    coeff_pm = zeros(nSeg, 4);  % [piston, tip, tilt, defocus] coefficients before final normalization

    switch lower(drift_definition)
        case {'rms_segment_loworder','bounded_segment_loworder'}
            for k = 1:nSeg
                mk = seg_id == k;
                if ~any(mk(:)), continue; end

                cx = seg_centers(k,1);
                cy = seg_centers(k,2);

                dx = (X - cx) / seg_radius_pix;
                dy = (Y - cy) / seg_radius_pix;
                rho2 = dx.^2 + dy.^2;
                defoc_shape = rho2 - mean(rho2(mk));

                c = randn(1,4);
                coeff_pm(k,:) = c;
                wfe_pm(mk) = c(1) + c(2)*dx(mk) + c(3)*dy(mk) + c(4)*defoc_shape(mk);
            end

        case {'rms_segment_piston','bounded_segment_piston'}
            piston_pm = randn(nSeg, 1);
            piston_pm = piston_pm - mean(piston_pm);
            coeff_pm(:,1) = piston_pm;
            for k = 1:nSeg
                mk = seg_id == k;
                if ~any(mk(:)), continue; end
                wfe_pm(mk) = piston_pm(k);
            end

        otherwise
            error('Unknown drift_definition: %s', drift_definition);
    end

    % If the user uses this to represent reflective surface motion, convert to
    % wavefront error before normalization/scoring.
    wfe_pm = surface_to_wavefront_factor * wfe_pm;

    % Global piston is not observable. Remove it before defining total WFE.
    wfe_pm(mask) = wfe_pm(mask) - mean(wfe_pm(mask), 'omitnan');

    switch lower(drift_definition)
        case {'rms_segment_loworder','rms_segment_piston'}
            denom = sqrt(mean(wfe_pm(mask).^2, 'omitnan'));
            scale_factor = actual_target_pm / max(denom, eps);
        case {'bounded_segment_loworder','bounded_segment_piston'}
            denom = max(abs(wfe_pm(mask)));
            scale_factor = actual_target_pm / max(denom, eps);
    end

    wfe_pm = scale_factor * wfe_pm;
    coeff_pm = surface_to_wavefront_factor * scale_factor * coeff_pm;

    % Re-remove tiny numerical/global piston after scaling.
    wfe_pm(mask) = wfe_pm(mask) - mean(wfe_pm(mask), 'omitnan');

    dphi_map = zeros(size(seg_id));
    dphi_map(mask) = 2*pi * (wfe_pm(mask) * 1e-12) / lambda_m;

    drift_pm_by_seg = nan(nSeg,1);
    for k = 1:nSeg
        mk = mask & (seg_id == k);
        if any(mk(:))
            drift_pm_by_seg(k) = mean(wfe_pm(mk), 'omitnan');
        end
    end

    info = struct();
    info.definition = string(drift_definition);
    info.actual_target_pm = actual_target_pm;
    info.scale_factor = scale_factor;
    info.coeff_pm = coeff_pm;
    info.coeff_rms_pm = sqrt(mean(coeff_pm(:).^2, 'omitnan'));
    info.wfe_rms_pm = sqrt(mean(wfe_pm(mask).^2, 'omitnan'));
    info.wfe_maxabs_pm = max(abs(wfe_pm(mask)));
    info.wfe_pv_pm = max(wfe_pm(mask)) - min(wfe_pm(mask));
end

function dphi = remove_piston(dphi, mask)
    mask = logical(mask) & isfinite(dphi);
    if ~any(mask(:)), return; end
    phasor_mean = angle(sum(exp(1i*dphi(mask))));
    dphi = wrap_phase(dphi - phasor_mean);
end

function y = wrap_phase(x)
    y = angle(exp(1i*x));
end

function pm = phase_rad_to_pm(phi_rad, lambda_m)
    pm = phi_rad * lambda_m / (2*pi) * 1e12;
end

function pm = phase_rms_to_pm(phi_rms_rad, lambda_m)
    pm = phase_rad_to_pm(phi_rms_rad, lambda_m);
end

function val = project_onto_true_drift_pm(dphi_rec, dphi_true, mask, lambda_m)
    mask = logical(mask) & isfinite(dphi_rec) & isfinite(dphi_true);
    if ~any(mask(:))
        val = NaN;
        return;
    end
    a = dphi_rec(mask);
    b = dphi_true(mask);
    denom = sum(b.^2);
    if denom <= eps
        val = NaN;
        return;
    end
    gain = sum(a .* b) / denom;
    projected_phase_rms = gain * sqrt(mean(b.^2));
    val = phase_rms_to_pm(projected_phase_rms, lambda_m);
end

function E_aligned = align_global_phase(E_ref, E, mask)
    mask = logical(mask) & abs(E_ref) > 0 & abs(E) > 0 & isfinite(real(E_ref)) & isfinite(real(E));
    if ~any(mask(:))
        E_aligned = E;
        return;
    end
    phi0 = angle(sum(conj(E_ref(mask)) .* E(mask)));
    E_aligned = E .* exp(-1i * phi0);
end

function rmse = phase_rmse(Ftrue, Frec, mask)
    mask = logical(mask) & abs(Ftrue) > 0 & abs(Frec) > 0;
    if ~any(mask(:))
        rmse = NaN;
        return;
    end
    phase_err = angle(conj(Ftrue(mask)) .* Frec(mask));
    rmse = sqrt(mean(phase_err.^2));
end

function r = masked_rms(x, mask)
    mask = logical(mask) & isfinite(x);
    if ~any(mask(:))
        r = NaN;
        return;
    end
    v = x(mask);
    r = sqrt(mean(v(:).^2));
end

function r = safe_rms(x)
    x = x(isfinite(x));
    if isempty(x)
        r = NaN;
    else
        r = sqrt(mean(x(:).^2));
    end
end

function c = safe_corr(a, b)
    a = a(:); b = b(:);
    good = isfinite(a) & isfinite(b);
    a = a(good); b = b(good);
    if numel(a) < 3 || std(a) <= 0 || std(b) <= 0
        c = NaN;
        return;
    end
    C = corrcoef(a,b);
    c = C(1,2);
end

%% ========================================================================
%  SNR AND SUPPORT HELPERS
%  ========================================================================

function snr = estimate_injected_focal_snr(I_clean, I_noisy, half_width)
%ESTIMATE_INJECTED_FOCAL_SNR Simulation-only SNR calibration.
% Signal is the central 21x21 mean. Noise is the standard deviation of the
% known injected noise realization in the corner 21x21.
%
% This is the right quantity for a simulation-control calibration plot. It
% should be close to linear on log(SNR) vs log(noise_level), with slope -1.
    I_clean = double(I_clean);
    I_noisy = double(I_noisy);
    [H,W] = size(I_clean);
    cv = round((H+1)/2);
    ch = round((W+1)/2);

    rr = max(1,cv-half_width):min(H,cv+half_width);
    cc = max(1,ch-half_width):min(W,ch+half_width);
    signal = mean(I_noisy(rr,cc), 'all');

    rr0 = 1:min(H, 2*half_width+1);
    cc0 = 1:min(W, 2*half_width+1);
    noise_realization = I_noisy(rr0,cc0) - I_clean(rr0,cc0);
    noise = std(noise_realization, 0, 'all');

    snr = signal / max(noise, eps);
end

function snr = estimate_single_image_corner_snr(I, half_width)
%ESTIMATE_SINGLE_IMAGE_CORNER_SNR Diagnostic only.
% This mimics the experimental single-frame estimate, but in a noiseless
% simulation it is contaminated by deterministic PSF structure in the corner.
% Therefore it can plateau and should not be used for the simulation SNR axis.
    I = double(I);
    [H,W] = size(I);
    cv = round((H+1)/2);
    ch = round((W+1)/2);

    rr = max(1,cv-half_width):min(H,cv+half_width);
    cc = max(1,ch-half_width):min(W,ch+half_width);
    signal = mean(I(rr,cc), 'all');

    rr0 = 1:min(H, 2*half_width+1);
    cc0 = 1:min(W, 2*half_width+1);
    noise = std(I(rr0,cc0), 0, 'all');

    snr = signal / max(noise, eps);
end

function CSg = correlation_support(E1_mask, E2_mask)
    big = conv2(double(flipud(fliplr(E1_mask))), double(E2_mask), 'same');
    CSg = big > 0;
end

function [seg_id, centers, X, Y] = make_hex_segment_aperture(N, a, num_rings)
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
    leak = norm(F(~support_mask)) / max(norm(F(:)), eps);
end

function r = relnorm(err, ref, mask)
    if nargin < 3 || isempty(mask)
        mask = true(size(ref));
    end
    mask = logical(mask);
    r = norm(err(mask)) / max(norm(ref(mask)), eps);
end

%% ========================================================================
%  DEVICE / STRUCT / GRAPHICS HELPERS
%  ========================================================================

function y = to_cpu(x)
    if isa(x, 'gpuArray')
        y = gather(x);
    else
        y = x;
    end
end

function S = gather_struct(S)
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

function safe_save_png(fig, filename)
    try
        exportgraphics(fig, filename, 'Resolution', 200);
    catch
        print(fig, filename, '-dpng', '-r200');
    end
end

function cleanup_session()
    try
        if exist('winsomnia','file') == 2
            winsomnia(false);
        end
    catch
    end
end
