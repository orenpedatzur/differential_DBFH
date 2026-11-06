%% sweep_dDBFH_vs_dOTF_perturbation_size_fixed.m
% Sweep perturbation size for one or more fixed noise levels and compare
% dDBFH vs dOTF using wrapped phase error only.
%
% Main plotted quantity:
%   wrapped phase RMSE relative to the ground-truth pupil field.
%
% Four plotted curves:
%   1) dDBFH reconstruction evaluated on the dOTF-valid region produced by
%      the dDBFH cap/string perturbation.
%   2) dOTF reconstruction using the same dDBFH cap/string perturbation.
%   3) dDBFH reconstruction evaluated on the dOTF-valid region produced by
%      the compact dOTF-favorable perturbation.
%   4) dOTF reconstruction using the compact dOTF-favorable perturbation.
%
% Important:
%   The dDBFH reconstruction is evaluated on the same valid mask as the dOTF
%   branch it is compared against. Do not compare full-support dDBFH error
%   against dOTF valid-only error; that is biased.
%
% Repeats:
%   nRealizations independent random realizations are run for every
%   noise-level / perturbation-size point. Each realization regenerates both
%   the segmented phase front and image noise.
%
% Run from the project root containing:
%   ImZ_from_ReZ/    diff_DBFH/    fftc/
%
% Based on demo_compare_dDBFH_vs_dOTF_HWO_v2.m
% 24 May 2026

clear; close all; clc;
winsomnia(true);

%% ========================================================================
%  PATHS
%  ========================================================================

script_path = mfilename('fullpath');
if isempty(script_path)
    project_root = pwd;
else
    project_root = fileparts(script_path);
end

addpath_if_dir(project_root);
addpath_if_dir(fullfile(project_root, 'ImZ_from_ReZ'));
addpath_if_dir(fullfile(project_root, 'diff_DBFH'));
addpath_if_dir(fullfile(project_root, 'fftc'));

%% ========================================================================
%  USER PARAMETERS
%  ========================================================================

zernike_nMax            = 2;
zernike_coeff_sigma_rad = 0.5;

% ------------------------------------------------------------------------
% Physical telescope / detector parameters
% ------------------------------------------------------------------------
% The sampling is set in the physically correct order:
%   1) choose telescope diameter, focal length, wavelength, detector pitch;
%   2) derive the pupil-plane FFT window L = lambda*f/p_cam;
%   3) derive the pupil-plane pixel size dx = L/N;
%   4) derive the aperture diameter and segment size in pupil pixels.
%
% This replaces the older logic where seg_radius_pix was chosen first and
% the focal-plane pixel pitch was only inferred afterward.

N                 = 2048;       % FFT grid size; also the modeled detector size [pixels]
D_m_inscribed     = 8.2;        % HWO EAC5 inscribed aperture diameter [m]
f_m               = 171;        % HWO effective focal length [m]
lambda_m          = 550e-9;     % reference wavelength [m]
cam_px_m          = 3.45e-6;     % detector pixel pitch [m]
num_rings         = 3;          % 3 rings -> 37 hex segments
gap_pix = 1 ; 

% Focal-plane sampling is fixed by the camera.
px_m_focal_plane = cam_px_m;

% Pupil-plane FFT window required to give this focal-plane sampling.
L_m_pupil        = lambda_m * f_m / px_m_focal_plane;
px_m_pupil_plane = L_m_pupil / N;

% Aperture diameter in the pupil-grid coordinates.
D_pix_inscribed = D_m_inscribed / px_m_pupil_plane;

aperture_diameter_factor = (2*num_rings + 1)*sqrt(3);


seg_radius_pix_exact = D_pix_inscribed / aperture_diameter_factor;
seg_radius_pix       = round(seg_radius_pix_exact);

% Recompute the effective physical sampling after rounding seg_radius_pix.
D_pix_inscribed_eff     = aperture_diameter_factor * seg_radius_pix;
px_m_pupil_plane_eff    = D_m_inscribed / D_pix_inscribed_eff;
L_m_pupil_eff           = N * px_m_pupil_plane_eff;
px_m_focal_plane_eff    = lambda_m * f_m / L_m_pupil_eff;
Q_pix_per_lambdaD_eff   = lambda_m * f_m / (D_m_inscribed * px_m_focal_plane_eff);
lambdaD_per_pix_eff     = 1 / Q_pix_per_lambdaD_eff;

% Backward-compatible variable names used later in the script.
Q = Q_pix_per_lambdaD_eff;

% Sweep variable: dDBFH cap/string depth in pixels.
% The dOTF compact blocker is regenerated at each point with the same number
% of masked pupil pixels as the dDBFH cap/string perturbation.
string_depth_list_pix = 10; %[1:1:10];
% string_depth_list_pix = [1:2:25];

% Noise levels. Leave scalar for the requested single-noise-level experiment.
noise_level_list  = [1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2];
% noise_level_list  = [1e-4];

% Independent random realizations per noise/size point.
nRealizations     = 20;
base_seed         = 1;

% Noise model: independent additive Gaussian noise on each PSF image,
% scaled by max(I_full(:)) for that perturbation size and realization.
use_noisy_mag1_for_dDBFH = true;

% Cross-term recovery controls for dDBFH.
d_theta           = 0.1;                % 0.1 is cleaner but much slower in sweeps
theta_list        = 0:d_theta:180-d_theta;
use_exact_ReZ     = false;               % false: ReZ = DeltaI/2 approximation
use_true_Z_solver = false;               % true: bypass KK and feed Z_true

% Solver controls copied from the working GPU demo.
useGPU            = true;
nsamp             = 20;
blockSize         = 1;
precond_cap       = 1e2;
lsqr_tol          = 1e-8;
lsqr_maxit        = 1e5;
K_threshold       = 1e-3;
supportScale      = 1;
crossWeight       = 100;

% dOTF controls.
dotf_opts = struct();
dotf_opts.copy_choice = 'autoByCoverage';
dotf_opts.exclude_block_pixels = true;

% Output controls.
show_plots        = true;
save_results      = true;
out_dir           = fullfile(project_root, 'results_dDBFH_vs_dOTF_noise_sweep');

if save_results && ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

if useGPU
    assert(gpuDeviceCount > 0, 'useGPU=true but no CUDA GPU is available. Set useGPU=false.');
    gd = gpuDevice;
    fprintf('Using GPU: %s, available memory %.2f GB\n', gd.Name, gd.AvailableMemory/1024^3);
end

%% ========================================================================
%  BUILD FIXED SEGMENTED APERTURE GEOMETRY
%  ========================================================================

[seg_id, seg_centers, X, Y] = make_hex_segment_aperture(N, seg_radius_pix, num_rings);
aperture = seg_id > 0;
nSeg = max(seg_id(:));

nNoise = numel(noise_level_list);
nSweep = numel(string_depth_list_pix);
nTotal = nNoise * nSweep * nRealizations;

NoiseLevel = nan(nTotal,1);
StringDepthPix = nan(nTotal,1);
Realization = nan(nTotal,1);
TrialSeed = nan(nTotal,1);
MaskedPixels = nan(nTotal,1);
CoverageSamePct = nan(nTotal,1);  % saved as diagnostic only; no coverage plot
CoverageOptPct  = nan(nTotal,1);  % saved as diagnostic only; no coverage plot

DBFH_same_PhaseRMSE_rad = nan(nTotal,1);
dOTF_same_PhaseRMSE_rad = nan(nTotal,1);
DBFH_opt_PhaseRMSE_rad  = nan(nTotal,1);
dOTF_opt_PhaseRMSE_rad  = nan(nTotal,1);
DBFH_full_PhaseRMSE_rad = nan(nTotal,1); % diagnostic only; not plotted

RelReZErr = nan(nTotal,1);
RelImZErr = nan(nTotal,1);
RelZErr   = nan(nTotal,1);
KKLineRelErr = nan(nTotal,1);

DBFH_LSQRFlag   = nan(nTotal,1);
DBFH_LSQRIter   = nan(nTotal,1);
DBFH_LSQRRelres = nan(nTotal,1);
DBFH_Time_s     = nan(nTotal,1);
dOTF_same_Time_s = nan(nTotal,1);
dOTF_opt_Time_s  = nan(nTotal,1);

failed = false(nTotal,1);
fail_message = strings(nTotal,1);

%% ========================================================================
%  SWEEP: NOISE LEVEL x PERTURBATION SIZE x RANDOM REALIZATION
%  ========================================================================

fprintf('\n===== Perturbation-size sweep: wrapped phase error only =====\n');
fprintf('noise_level_list = %s\n', mat2str(noise_level_list));
fprintf('depth list       = %s pix\n', mat2str(string_depth_list_pix));
fprintf('nRealizations    = %d\n', nRealizations);

row = 0;
for iNoise = 1:nNoise
    noise_level = noise_level_list(iNoise);

    for iSize = 1:nSweep
        string_depth_pix = string_depth_list_pix(iSize);
%%
        % Geometry depends only on perturbation size, not realization.
        geom = make_dbfh_cap_string(seg_id, seg_centers, X, Y, aperture, string_depth_pix, gap_pix);
        supp1_dbfh  = geom.supp1;
        supp2_dbfh  = geom.supp2;
        supp12_dbfh = geom.supp12;
        nMask       = nnz(supp2_dbfh);

        if nMask < 1
            error('The dDBFH cap/string perturbation has zero pixels. Increase string_depth_pix.');
        end

        supp2_dotf_opt = make_codona_blocker(supp12_dbfh, X, Y, nMask);
%%
        for iReal = 1:nRealizations
            %%
            row = row + 1;
            trial_seed = base_seed + 100000*iNoise + 1000*iSize + iReal;

            NoiseLevel(row) = noise_level;
            StringDepthPix(row) = string_depth_pix;
            Realization(row) = iReal;
            TrialSeed(row) = trial_seed;
            MaskedPixels(row) = nMask;

            fprintf('\n=================================================================\n');
            fprintf('Noise %d/%d = %.3e | size %d/%d = %g pix | realization %d/%d\n', ...
                iNoise, nNoise, noise_level, iSize, nSweep, string_depth_pix, iReal, nRealizations);
            fprintf('Masked pixels = %d\n', nMask);
            fprintf('=================================================================\n');

            try
                rng(trial_seed, 'twister');

                %% Random truth field for this realization
                E = make_random_segmented_field(seg_id, seg_centers, X, Y, seg_radius_pix, nSeg, zernike_nMax, zernike_coeff_sigma_rad);

                %% Truth fields for this comparison support
                E12      = E .* supp12_dbfh;
                E1_dbfh  = E .* supp1_dbfh;
                E2_dbfh  = E .* supp2_dbfh;

                E1_dotf = E12;
                E1_dotf(supp2_dotf_opt) = 0;

                %% Forward images
                A12 = fft2c(E12);
                I12_clean = abs(A12).^2;

                A1_dbfh = fft2c(E1_dbfh);
                A2_dbfh = fft2c(E2_dbfh);
                I1_dbfh_clean = abs(A1_dbfh).^2;

                A1_dotf = fft2c(E1_dotf);
                I1_dotf_clean = abs(A1_dotf).^2;

                noise_scale = max(I12_clean(:));
                I12     = I12_clean     + noise_level * noise_scale * randn(size(I12_clean));
                I1_dbfh = I1_dbfh_clean + noise_level * noise_scale * randn(size(I12_clean));
                I1_dotf = I1_dotf_clean + noise_level * noise_scale * randn(size(I12_clean));

                %% dDBFH cross-term construction
                Z_true    = conj(A1_dbfh) .* A2_dbfh;
                ReZ_exact = real(Z_true);
                ImZ_true  = imag(Z_true);

                DeltaI_dbfh = I12 - I1_dbfh;
                if use_exact_ReZ
                    ReZ_meas = ReZ_exact;
                else
                    % Differential approximation. The |A2|^2 bias is intentionally
                    % left in place; do not subtract truth information here.
                    ReZ_meas = 0.5 * DeltaI_dbfh;
                end

                CSg = correlation_support(supp1_dbfh, supp2_dbfh);

                if use_true_Z_solver
                    ImZ_rec = ImZ_true;
                    Z_solver = Z_true;
                    kk = struct();
                else
                    if ~exist('recover_ImZ_by_cartesian_KK', 'file')
                        error(['recover_ImZ_by_cartesian_KK not found. Check ImZ_from_ReZ path ', ...
                               'or set use_true_Z_solver=true for solver-only testing.']);
                    end
                    [ImZ_rec, kk] = recover_ImZ_by_cartesian_KK( ...
                        ReZ_meas, CSg, theta_list, ...
                        'InterpMethod', 'cubic', ...
                        'trueImZ', ImZ_true);
                    Z_solver = ReZ_meas + 1i*ImZ_rec;
                end

                RelReZErr(row) = relnorm(ReZ_meas - ReZ_exact, ReZ_exact, true(size(ReZ_exact)));
                RelImZErr(row) = relnorm(ImZ_rec  - ImZ_true,  ImZ_true,  true(size(ImZ_true)));
                RelZErr(row)   = relnorm(Z_solver - Z_true,    Z_true,    true(size(Z_true)));
                if isfield(kk, 'line_rel_err')
                    KKLineRelErr(row) = kk.line_rel_err;
                end

                fprintf('relative ReZ / ImZ / Z error = %.3e / %.3e / %.3e\n', ...
                    RelReZErr(row), RelImZErr(row), RelZErr(row));

                if use_noisy_mag1_for_dDBFH
                    I_mag_dbfh = I1_dbfh;
                else
                    I_mag_dbfh = I1_dbfh_clean;
                end
                mag1 = sqrt(max(I_mag_dbfh, 0));

                %% diff-DBFH solve
                dbfh = run_dDBFH_hard_anchor_solver( ...
                    mag1, Z_solver, supp1_dbfh, supp2_dbfh, E12, useGPU, ...
                    nsamp, blockSize, precond_cap, lsqr_tol, lsqr_maxit, ...
                    K_threshold, supportScale, crossWeight);

                DBFH_LSQRFlag(row)   = dbfh.flag;
                DBFH_LSQRIter(row)   = dbfh.iter;
                DBFH_LSQRRelres(row) = dbfh.relres;
                DBFH_Time_s(row)     = dbfh.solve_time;

                %% dOTF branches
                fprintf('\n--- dOTF using same dDBFH cap/string perturbation ---\n');
                dotf_same = run_dOTF_blocked_probe(I12, I1_dbfh, E12, supp12_dbfh, supp2_dbfh, X, Y, dotf_opts);

                fprintf('\n--- dOTF using compact edge/corner perturbation ---\n');
                dotf_opt = run_dOTF_blocked_probe(I12, I1_dotf, E12, supp12_dbfh, supp2_dotf_opt, X, Y, dotf_opts);

                dOTF_same_Time_s(row) = dotf_same.solve_time;
                dOTF_opt_Time_s(row)  = dotf_opt.solve_time;

                %% Fair phase-only metrics
                % dDBFH is evaluated only where the matching dOTF branch is valid.
                DBFH_full_PhaseRMSE_rad(row) = phase_rmse(E12, dbfh.E_rec_scaled, supp12_dbfh);
                DBFH_same_PhaseRMSE_rad(row) = phase_rmse(E12, dbfh.E_rec_scaled, dotf_same.valid);
                dOTF_same_PhaseRMSE_rad(row) = phase_rmse(E12, dotf_same.E_rec_scaled, dotf_same.valid);
                DBFH_opt_PhaseRMSE_rad(row)  = phase_rmse(E12, dbfh.E_rec_scaled, dotf_opt.valid);
                dOTF_opt_PhaseRMSE_rad(row)  = phase_rmse(E12, dotf_opt.E_rec_scaled, dotf_opt.valid);

                CoverageSamePct(row) = 100 * nnz(dotf_same.valid) / nnz(supp12_dbfh);
                CoverageOptPct(row)  = 100 * nnz(dotf_opt.valid)  / nnz(supp12_dbfh);

                fprintf('\nWrapped phase RMSE curves at this trial:\n');
                fprintf('  dDBFH on same-cap dOTF-valid region = %.3e rad\n', DBFH_same_PhaseRMSE_rad(row));
                fprintf('  dOTF  on same-cap dOTF-valid region = %.3e rad\n', dOTF_same_PhaseRMSE_rad(row));
                fprintf('  dDBFH on opt-dOTF valid region      = %.3e rad\n', DBFH_opt_PhaseRMSE_rad(row));
                fprintf('  dOTF  on opt-dOTF valid region      = %.3e rad\n', dOTF_opt_PhaseRMSE_rad(row));

                clear dbfh dotf_same dotf_opt I12 I1_dbfh I1_dotf I12_clean I1_dbfh_clean I1_dotf_clean E

            catch ME
                failed(row) = true;
                fail_message(row) = string(ME.message);
                warning('Trial failed: noise %.3e, depth %g, realization %d: %s', ...
                    noise_level, string_depth_pix, iReal, ME.message);
            end
        end
    end
end

%% ========================================================================
%  TRIAL TABLE AND SUMMARY TABLE
%  ========================================================================

metrics_trial = table( ...
    NoiseLevel, StringDepthPix, Realization, TrialSeed, MaskedPixels, ...
    DBFH_same_PhaseRMSE_rad, dOTF_same_PhaseRMSE_rad, DBFH_opt_PhaseRMSE_rad, dOTF_opt_PhaseRMSE_rad, DBFH_full_PhaseRMSE_rad, ...
    CoverageSamePct, CoverageOptPct, ...
    RelReZErr, RelImZErr, RelZErr, KKLineRelErr, ...
    DBFH_LSQRFlag, DBFH_LSQRIter, DBFH_LSQRRelres, DBFH_Time_s, dOTF_same_Time_s, dOTF_opt_Time_s, ...
    failed, fail_message);

disp(metrics_trial);

summary_phase = summarize_phase_trials(metrics_trial, noise_level_list, string_depth_list_pix);
disp(summary_phase);

if save_results
    writetable(metrics_trial, fullfile(out_dir, 'dDBFH_vs_dOTF_phase_trials.csv'));
    writetable(summary_phase, fullfile(out_dir, 'dDBFH_vs_dOTF_phase_summary_median_iqr.csv'));
    save(fullfile(out_dir, 'dDBFH_vs_dOTF_phase_sweep_results.mat'), ...
        'metrics_trial', 'summary_phase', 'noise_level_list', 'string_depth_list_pix', ...
        'nRealizations', 'N', 'seg_radius_pix', 'num_rings', 'gap_pix', ...
        'zernike_nMax', 'zernike_coeff_sigma_rad', ...
        'use_noisy_mag1_for_dDBFH', 'use_exact_ReZ', 'use_true_Z_solver', ...
        'K_threshold', 'crossWeight', '-v7.3');
end

%% ========================================================================
%  PHASE-ONLY PLOTS
%  ========================================================================

if show_plots
    % plot_phase_vs_size_summary(summary_phase, noise_level_list, nRealizations, out_dir, save_results);
    plot_phase_vs_noise_summary(summary_phase, string_depth_list_pix, nRealizations, out_dir, save_results)
end


winsomnia(false);

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function geom = make_dbfh_cap_string(seg_id, seg_centers, X, Y, aperture, string_depth_pix, gap_pix)
%MAKE_DBFH_CAP_STRING Build the peripheral cap/string perturbation for dDBFH.
[~, top_seg] = max(seg_centers(:,2));
top_mask = seg_id == top_seg;

c_top = seg_centers(top_seg,:);
n_out = c_top / norm(c_top);
S = X*n_out(1) + Y*n_out(2);
S_top = S(top_mask);
s_edge = max(S_top);

cut_E2  = s_edge - string_depth_pix;
cut_gap = cut_E2 - gap_pix;

supp2_dbfh = top_mask & (S >= cut_E2);
gap_mask   = top_mask & (S >= cut_gap) & (S < cut_E2);
supp1_dbfh = aperture & ~supp2_dbfh & ~gap_mask;
supp12_dbfh = supp1_dbfh | supp2_dbfh;

geom = struct();
geom.supp1 = supp1_dbfh;
geom.supp2 = supp2_dbfh;
geom.supp12 = supp12_dbfh;
geom.gap_mask = gap_mask;
geom.top_seg = top_seg;
geom.n_out = n_out;
geom.s_edge = s_edge;
geom.cut_E2 = cut_E2;
geom.cut_gap = cut_gap;
end

function E = make_random_segmented_field(seg_id, seg_centers, X, Y, seg_radius_pix, nSeg, zernike_nMax, coeff_sigma_rad)
%MAKE_RANDOM_SEGMENTED_FIELD Random segmented phase front from per-segment Zernikes.
%
% E = make_random_segmented_field(..., zernike_nMax, coeff_sigma_rad)
%
% zernike_nMax:
%   maximum radial Zernike order included on each segment.
%   0 = piston only
%   1 = piston + tip/tilt
%   2 = piston + tip/tilt + defocus/astigmatism
%   3 = + coma/trefoil
%   etc.
%
% coeff_sigma_rad:
%   scalar or vector of random coefficient standard deviations in radians.
%   If scalar, the same sigma is used for every included mode.
%   If vector, it must have one entry per generated real Zernike mode.
%
% Basis convention:
%   Real Zernikes on normalized local segment coordinates:
%       m = 0       : R_n^0(r)
%       m > 0, cos : R_n^m(r) cos(m theta)
%       m > 0, sin : R_n^m(r) sin(m theta)
%   Each mode is mean-subtracted except piston, then RMS-normalized on the
%   actual hexagonal segment pixels. Therefore coeff_sigma_rad is directly a
%   per-mode phase scale in radians over the segment mask.

if nargin < 7 || isempty(zernike_nMax)
    zernike_nMax = 2;
end
if nargin < 8 || isempty(coeff_sigma_rad)
    coeff_sigma_rad = 1.0;
end
if zernike_nMax < 0 || zernike_nMax ~= floor(zernike_nMax)
    error('zernike_nMax must be a nonnegative integer.');
end

modes = make_real_zernike_mode_list(zernike_nMax);
nModes = numel(modes);

if isscalar(coeff_sigma_rad)
    coeff_sigma_rad = coeff_sigma_rad * ones(nModes,1);
else
    coeff_sigma_rad = coeff_sigma_rad(:);
    if numel(coeff_sigma_rad) ~= nModes
        error('coeff_sigma_rad must be scalar or have one entry per mode. nModes = %d.', nModes);
    end
end

E = zeros(size(seg_id));

for k = 1:nSeg
    mk = seg_id == k;
    if ~any(mk(:)), continue; end

    cx = seg_centers(k,1);
    cy = seg_centers(k,2);

    dx = (X - cx) / seg_radius_pix;
    dy = (Y - cy) / seg_radius_pix;
    r = hypot(dx, dy);
    th = atan2(dy, dx);

    phi = zeros(size(seg_id));

    for j = 1:nModes
        n = modes(j).n;
        m = modes(j).m;
        kind = modes(j).kind;

        Z = real_zernike_nm(n, m, kind, r, th);

        if ~(n == 0 && m == 0)
            Z = Z - mean(Z(mk));
        end

        zrms = sqrt(mean(Z(mk).^2));
        if zrms <= eps || ~isfinite(zrms)
            continue;
        end
        Z = Z / zrms;

        coeff = coeff_sigma_rad(j) * randn();
        phi(mk) = phi(mk) + coeff * Z(mk);
    end

    E(mk) = exp(1i * phi(mk));
end
end

function modes = make_real_zernike_mode_list(nMax)
%MAKE_REAL_ZERNIKE_MODE_LIST Real Zernike modes through radial order nMax.
% Order is piston, then increasing n, then m. For m>0, cos then sin.

modes = struct('n', {}, 'm', {}, 'kind', {});
for n = 0:nMax
    for m = 0:n
        if mod(n - m, 2) ~= 0
            continue;
        end
        if m == 0
            modes(end+1) = struct('n', n, 'm', m, 'kind', 'm0'); %#ok<AGROW>
        else
            modes(end+1) = struct('n', n, 'm', m, 'kind', 'cos'); %#ok<AGROW>
            modes(end+1) = struct('n', n, 'm', m, 'kind', 'sin'); %#ok<AGROW>
        end
    end
end
end

function Z = real_zernike_nm(n, m, kind, r, th)
%REAL_ZERNIKE_NM Real-valued Zernike mode evaluated on r, theta.

R = zernike_radial(n, m, r);

switch lower(kind)
    case 'm0'
        Z = R;
    case 'cos'
        Z = R .* cos(m * th);
    case 'sin'
        Z = R .* sin(m * th);
    otherwise
        error('Unknown Zernike kind: %s', kind);
end
end

function R = zernike_radial(n, m, r)
%ZERNIKE_RADIAL Radial polynomial R_n^m(r), using the standard definition.

if m < 0 || m > n || mod(n - m, 2) ~= 0
    R = zeros(size(r));
    return;
end

R = zeros(size(r));
for s = 0:((n - m)/2)
    logc = gammaln(n - s + 1) ...
         - gammaln(s + 1) ...
         - gammaln((n + m)/2 - s + 1) ...
         - gammaln((n - m)/2 - s + 1);
    c = (-1)^s * exp(logc);
    R = R + c * r.^(n - 2*s);
end
end

function dbfh = run_dDBFH_hard_anchor_solver(mag1, Z_solver, supp1_dbfh, supp2_dbfh, E12, useGPU, ...
    nsamp, blockSize, precond_cap, lsqr_tol, lsqr_maxit, K_threshold, supportScale, crossWeight)
%RUN_DBFH_HARD_ANCHOR_SOLVER Solve the dDBFH hard-anchor system and return aligned field.

fprintf('\n--- dDBFH hard-anchor solve ---\n');

if useGPU
    mag1_sys  = gpuArray(mag1);
    Z_sys     = gpuArray(Z_solver);
    supp1_sys = gpuArray(supp1_dbfh);
    supp2_sys = gpuArray(supp2_dbfh);

    sys = diff_dbfh_prepare_system_gpu( ...
        mag1_sys, Z_sys, supp1_sys, supp2_sys, ...
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
        mag1, Z_solver, supp1_dbfh, supp2_dbfh, ...
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

fprintf('sys.n = %d = nX1 %d + nE2 %d; valid K pixels = %d\n', sys.n, sys.nX1, sys.nE2, sys.nK);
fprintf('K threshold / crossWeight = %.1e / %.3g\n', K_threshold, crossWeight);

fprintf('Hutchinson right preconditioner...\n');
d = hutchinson_diag_AtA(sys, nsamp, blockSize, useGPU);
d = d(:);
d_cpu = to_cpu(d);
dpos = d_cpu(d_cpu > 0 & isfinite(d_cpu));
assert(~isempty(dpos), 'Hutchinson diagonal has no positive finite entries.');

dfloor = 1e-8 * median(dpos);
p0 = 1 ./ sqrt(max(d, dfloor));
p0_cpu = to_cpu(p0);
p0pos = p0_cpu(isfinite(p0_cpu) & p0_cpu > 0);
p_med = median(p0pos);
p_cap = precond_cap * p_med;
p = min(p0, p_cap);

Afun = make_Afun_lsqr_stacked_diagprecond(sys, p);
bstack = [real(sys.b); imag(sys.b)];

fprintf('LSQR solve...\n');
tic;
[ustack, dbfh_flag, dbfh_relres, dbfh_iter, dbfh_resvec] = lsqr(Afun, bstack, lsqr_tol, lsqr_maxit);
solve_time = toc;

ur = ustack(1:sys.n);
ui = ustack(sys.n+1:end);
xvec = p .* (ur + 1i*ui);

if useGPU
    dbfh_unpack = diff_dbfh_unpack_solution_gpu(sys, xvec, 'normalizeX1', false);
    dbfh_unpack = gather_struct(dbfh_unpack);
else
    dbfh_unpack = diff_dbfh_unpack_solution(sys, xvec, 'normalizeX1', false);
end

E_dbfh_raw = dbfh_unpack.E;
[~, E_dbfh_scaled, alpha_dbfh] = align_complex_scalar(E12, E_dbfh_raw, abs(E12) > 0);

dbfh = struct();
dbfh.E_rec_raw = E_dbfh_raw;
dbfh.E_rec_scaled = E_dbfh_scaled;
dbfh.alpha = alpha_dbfh;
dbfh.flag = dbfh_flag;
dbfh.relres = double(to_cpu(dbfh_relres));
dbfh.iter = dbfh_iter;
dbfh.resvec = to_cpu(dbfh_resvec);
dbfh.solve_time = solve_time;
dbfh.leak1 = dbfh_unpack.leak1;
dbfh.leak2 = dbfh_unpack.leak2;
dbfh.cross_relres = dbfh_unpack.cross_relres;

fprintf('dDBFH flag/relres/iter/time = %d / %.3e / %d / %.2f s\n', ...
    dbfh.flag, dbfh.relres, dbfh.iter, dbfh.solve_time);
end

function dotf = run_dOTF_blocked_probe(I_full, I_prtb, E12, supp12_dbfh, block_mask, X, Y, opts)
%RUN_DOTF_BLOCKED_PROBE Codona-style two-image dOTF extraction.
%
% I_full    : PSF of the unperturbed full pupil E12.
% I_prtb    : PSF after inserting the perturbing opaque block/mask.
% E12       : complex truth field on the unperturbed pupil, for metrics only.
% block_mask: pupil pixels blocked in the perturbed capture.

if nargin < 8 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'copy_choice') || isempty(opts.copy_choice)
    opts.copy_choice = 'autoByCoverage';
end
if ~isfield(opts, 'exclude_block_pixels') || isempty(opts.exclude_block_pixels)
    opts.exclude_block_pixels = true;
end

I_full = double(I_full);
I_prtb = double(I_prtb);
E12 = double(E12);
supp12_dbfh = logical(supp12_dbfh);
block_mask = logical(block_mask);

assert(isequal(size(I_full), size(I_prtb), size(E12), size(supp12_dbfh), size(block_mask)), ...
    'I_full, I_prtb, E12, supp12_dbfh, and block_mask must have identical size.');

tic;
dO = ifft2c(I_prtb - I_full);

x0 = round(mean(X(block_mask)));
y0 = round(mean(Y(block_mask)));

CS_full_block  = correlation_support(supp12_dbfh, block_mask);
CS_block_full  = correlation_support(block_mask, supp12_dbfh);
CS_block_block = correlation_support(block_mask, block_mask);

cand_plus = extract_dOTF_blocked_copy(dO, E12, supp12_dbfh, block_mask, X, Y, ...
    x0, y0, CS_full_block, CS_block_full, CS_block_block, 'plus', opts.exclude_block_pixels);
cand_minus = extract_dOTF_blocked_copy(dO, E12, supp12_dbfh, block_mask, X, Y, ...
    x0, y0, CS_full_block, CS_block_full, CS_block_block, 'minus', opts.exclude_block_pixels);

switch lower(opts.copy_choice)
    case 'plus'
        chosen = cand_plus;
    case 'minus'
        chosen = cand_minus;
    case 'autobycoverage'
        if nnz(cand_minus.valid) > nnz(cand_plus.valid)
            chosen = cand_minus;
        else
            chosen = cand_plus;
        end
    otherwise
        error('Unknown dOTF copy_choice: %s', opts.copy_choice);
end

solve_time = toc;

if nnz(chosen.valid) == 0
    error('dOTF extraction produced an empty valid mask. Check support geometry/FFT convention.');
end

dotf = chosen;
dotf.method = 'dOTF blocked probe';
dotf.dOTF = dO;
dotf.I_full = I_full;
dotf.I_prtb = I_prtb;
dotf.block_mask = block_mask;
dotf.CS_full_block = CS_full_block;
dotf.CS_block_full = CS_block_full;
dotf.CS_block_block = CS_block_block;
dotf.solve_time = solve_time;
dotf.anchor_xy = [x0, y0];
dotf.candidate_plus = strip_heavy_candidate(cand_plus);
dotf.candidate_minus = strip_heavy_candidate(cand_minus);

fprintf('selected copy = %s\n', dotf.copy_name);
fprintf('masked pixels = %d; valid coverage = %.2f %%\n', nnz(block_mask), 100*nnz(dotf.valid)/nnz(supp12_dbfh));
fprintf('wrapped phase RMSE = %.3e rad\n', dotf.phase_rmse);
end

function cand = extract_dOTF_blocked_copy(dO, E12, supp12_dbfh, block_mask, X, Y, ...
    x0, y0, CS_full_block, CS_block_full, CS_block_block, mode, exclude_block_pixels)
%EXTRACT_DOTF_BLOCKED_COPY Extract one of the two shifted dOTF pupil copies.

[Nr,Nc] = size(dO);
cx = round((Nc+1)/2);
cy = round((Nr+1)/2);

switch lower(mode)
    case 'plus'
        % IFT[A_full^* A_block] sampled at s = x_block - x_pupil.
        sx = x0 - X;
        sy = y0 - Y;
        copy_name = 'plus: conj IFT[A_full^* A_block], s=x0-x';
        use_conj = true;
        ownCS = CS_full_block;
        otherCS = CS_block_full;
    case 'minus'
        % IFT[A_block^* A_full] sampled at s = x_pupil - x_block.
        sx = X - x0;
        sy = Y - y0;
        copy_name = 'minus: IFT[A_block^* A_full], s=x-x0';
        use_conj = false;
        ownCS = CS_block_full;
        otherCS = CS_full_block;
    otherwise
        error('Unknown dOTF copy mode: %s', mode);
end

col = cx + sx;
row = cy - sy;
in = row >= 1 & row <= Nr & col >= 1 & col <= Nc;

idx = zeros(Nr,Nc);
idx(in) = sub2ind([Nr,Nc], row(in), col(in));

raw = complex(zeros(Nr,Nc));
own_at = false(Nr,Nc);
other_at = false(Nr,Nc);
mini_at = false(Nr,Nc);

raw_vals = dO(idx(in));
if use_conj
    raw_vals = conj(raw_vals);
end
raw(in) = raw_vals;
own_at(in) = ownCS(idx(in));
other_at(in) = otherCS(idx(in));
mini_at(in) = CS_block_block(idx(in));

valid = supp12_dbfh & in & own_at & ~other_at & ~mini_at & ...
    isfinite(real(raw)) & isfinite(imag(raw));

if exclude_block_pixels
    valid = valid & ~block_mask;
end

if nnz(valid) > 0
    [~, E_rec_scaled, alpha] = align_complex_scalar(E12, raw, valid);
    phase_rmse_val = phase_rmse(E12, E_rec_scaled, valid);
else
    E_rec_scaled = raw;
    alpha = NaN;
    phase_rmse_val = Inf;
end

cand = struct();
cand.copy_name = copy_name;
cand.x0 = x0;
cand.y0 = y0;
cand.raw = raw;
cand.E_rec_scaled = E_rec_scaled;
cand.alpha = alpha;
cand.valid = valid;
cand.valid_ownCS = own_at;
cand.valid_otherCS = other_at;
cand.valid_miniCS = mini_at;
cand.phase_rmse = phase_rmse_val;
end

function B = make_codona_blocker(supp12_dbfh, X, Y, nMask)
%MAKE_CODONA_BLOCKER dOTF-favorable compact block near an outer corner.
idx = find(supp12_dbfh);
assert(nMask <= numel(idx), 'Requested blocker has more pixels than the support.');

r2 = X(idx).^2 + Y(idx).^2;
[~, j0] = max(r2);
x0 = X(idx(j0));
y0 = Y(idx(j0));

d2 = (X(idx) - x0).^2 + (Y(idx) - y0).^2;
[~, ord] = sort(d2, 'ascend');
sel = idx(ord(1:nMask));

B = false(size(supp12_dbfh));
B(sel) = true;
end

function [Frec_phase, Frec_scaled, alpha] = align_complex_scalar(Ftrue, Frec, mask)
mask = logical(mask) & abs(Ftrue) > 0 & abs(Frec) > 0;
if ~any(mask(:))
    Frec_phase = Frec;
    Frec_scaled = Frec;
    alpha = NaN;
    return;
end
phi0 = angle(sum(conj(Ftrue(mask)) .* Frec(mask)));
Frec_phase = Frec .* exp(-1i*phi0);
alpha = sum(conj(Frec(mask)) .* Ftrue(mask)) / max(sum(abs(Frec(mask)).^2), eps);
Frec_scaled = alpha .* Frec;
end

function CSg = correlation_support(E1_mask, E2_mask)
%CORRELATION_SUPPORT Support of IFT[A1^* A2].
big = conv2(double(flipud(fliplr(E1_mask))), double(E2_mask), 'same');
CSg = big > 0;
end

function [seg_id, centers, X, Y] = make_hex_segment_aperture(N, a, num_rings)
%MAKE_HEX_SEGMENT_APERTURE Build a segmented hex aperture.
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

function F = fft2c(u)
F = fftshift(fft2(ifftshift(u))) / sqrt(numel(u));
end

function u = ifft2c(F)
u = fftshift(ifft2(ifftshift(F))) * sqrt(numel(F));
end

function r = relnorm(err, ref, mask)
if nargin < 3 || isempty(mask)
    mask = true(size(ref));
end
mask = logical(mask) & isfinite(real(err)) & isfinite(imag(err)) & ...
    isfinite(real(ref)) & isfinite(imag(ref));
r = norm(err(mask)) / max(norm(ref(mask)), eps);
end

function rmse = phase_rmse(Ftrue, Frec, mask)
%PHASE_RMSE Wrapped phase RMSE relative to ground truth.
mask = logical(mask) & abs(Ftrue) > 0 & abs(Frec) > 0 & ...
    isfinite(real(Frec)) & isfinite(imag(Frec));
if ~any(mask(:))
    rmse = NaN;
    return;
end
phase_err = angle(conj(Ftrue(mask)) .* Frec(mask));
rmse = sqrt(mean(phase_err.^2));
end

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

function C = strip_heavy_candidate(C)
if isfield(C, 'raw'), C = rmfield(C, 'raw'); end
if isfield(C, 'E_rec_scaled'), C = rmfield(C, 'E_rec_scaled'); end
end

function addpath_if_dir(p)
if exist(p, 'dir')
    addpath(p);
end
end

function summary_phase = summarize_phase_trials(T, noise_level_list, string_depth_list_pix)
%SUMMARIZE_PHASE_TRIALS Median/IQR of wrapped phase RMSE for each noise/size point.

nNoise = numel(noise_level_list);
nSweep = numel(string_depth_list_pix);
nRows = nNoise * nSweep;

NoiseLevel = nan(nRows,1);
StringDepthPix = nan(nRows,1);
MaskedPixelsMedian = nan(nRows,1);
NTrialsRequested = nan(nRows,1);
NTrialsValid = nan(nRows,1);
NTrialsFailed = nan(nRows,1);

DBFH_same_med = nan(nRows,1); DBFH_same_q25 = nan(nRows,1); DBFH_same_q75 = nan(nRows,1);
dOTF_same_med = nan(nRows,1); dOTF_same_q25 = nan(nRows,1); dOTF_same_q75 = nan(nRows,1);
DBFH_opt_med  = nan(nRows,1); DBFH_opt_q25  = nan(nRows,1); DBFH_opt_q75  = nan(nRows,1);
dOTF_opt_med  = nan(nRows,1); dOTF_opt_q25  = nan(nRows,1); dOTF_opt_q75  = nan(nRows,1);
DBFH_full_med = nan(nRows,1); DBFH_full_q25 = nan(nRows,1); DBFH_full_q75 = nan(nRows,1);

row = 0;
for iNoise = 1:nNoise
    for iSize = 1:nSweep
        row = row + 1;
        nl = noise_level_list(iNoise);
        sd = string_depth_list_pix(iSize);
        sel = T.NoiseLevel == nl & T.StringDepthPix == sd;

        NoiseLevel(row) = nl;
        StringDepthPix(row) = sd;
        MaskedPixelsMedian(row) = nan_median_local(T.MaskedPixels(sel));
        NTrialsRequested(row) = nnz(sel);
        NTrialsFailed(row) = nnz(sel & T.failed);

        valid = sel & ~T.failed;
        NTrialsValid(row) = nnz(valid);

        [DBFH_same_med(row), DBFH_same_q25(row), DBFH_same_q75(row)] = median_iqr(T.DBFH_same_PhaseRMSE_rad(valid));
        [dOTF_same_med(row), dOTF_same_q25(row), dOTF_same_q75(row)] = median_iqr(T.dOTF_same_PhaseRMSE_rad(valid));
        [DBFH_opt_med(row),  DBFH_opt_q25(row),  DBFH_opt_q75(row)]  = median_iqr(T.DBFH_opt_PhaseRMSE_rad(valid));
        [dOTF_opt_med(row),  dOTF_opt_q25(row),  dOTF_opt_q75(row)]  = median_iqr(T.dOTF_opt_PhaseRMSE_rad(valid));
        [DBFH_full_med(row), DBFH_full_q25(row), DBFH_full_q75(row)] = median_iqr(T.DBFH_full_PhaseRMSE_rad(valid));
    end
end

summary_phase = table( ...
    NoiseLevel, StringDepthPix, MaskedPixelsMedian, NTrialsRequested, NTrialsValid, NTrialsFailed, ...
    DBFH_same_med, DBFH_same_q25, DBFH_same_q75, ...
    dOTF_same_med, dOTF_same_q25, dOTF_same_q75, ...
    DBFH_opt_med, DBFH_opt_q25, DBFH_opt_q75, ...
    dOTF_opt_med, dOTF_opt_q25, dOTF_opt_q75, ...
    DBFH_full_med, DBFH_full_q25, DBFH_full_q75);
end

function [med, q25, q75] = median_iqr(x)
x = x(:);
x = x(isfinite(x));
if isempty(x)
    med = NaN; q25 = NaN; q75 = NaN;
    return;
end
med = median(x);
q25 = quantile_local(x, 0.25);
q75 = quantile_local(x, 0.75);
end

function m = nan_median_local(x)
x = x(:);
x = x(isfinite(x));
if isempty(x)
    m = NaN;
else
    m = median(x);
end
end

function q = quantile_local(x, p)
%QUANTILE_LOCAL Toolbox-free linear-interpolated quantile.
x = sort(x(:));
x = x(isfinite(x));
n = numel(x);
if n == 0
    q = NaN;
elseif n == 1
    q = x;
else
    pos = 1 + (n-1)*p;
    lo = floor(pos);
    hi = ceil(pos);
    if lo == hi
        q = x(lo);
    else
        q = x(lo) + (pos-lo) * (x(hi) - x(lo));
    end
end
end

function plot_phase_vs_size_summary(S, noise_level_list, nRealizations, out_dir, save_results)

% labels = {'dDBFH | cap/string valid region', ...
%           'dOTF | cap/string perturbation', ...
%           'dDBFH | dOTF-opt valid region', ...
%           'dOTF | dOTF-opt perturbation'};
labels = {'dDBFH | string-cut perturbation', ...
    'dOTF | string-cut perturbation', ...
    'dOTF | dOTF perturbation'};

for iNoise = 1:numel(noise_level_list)
    nl = noise_level_list(iNoise);
    sel = S.NoiseLevel == nl;
    x = S.MaskedPixelsMedian(sel);

    figure('Color','w','Name',sprintf('Wrapped phase RMSE, noise %.1e', nl), ...
        'Position',[100 100 950 620]);
    hold on;

    plot_iqr_curve(x, S.DBFH_same_med(sel), S.DBFH_same_q25(sel), S.DBFH_same_q75(sel), '-o');
    plot_iqr_curve(x, S.dOTF_same_med(sel), S.dOTF_same_q25(sel), S.dOTF_same_q75(sel), '-s');
    % plot_iqr_curve(x, S.DBFH_opt_med(sel),  S.DBFH_opt_q25(sel),  S.DBFH_opt_q75(sel),  '-^');
    plot_iqr_curve(x, S.dOTF_opt_med(sel),  S.dOTF_opt_q25(sel),  S.dOTF_opt_q75(sel),  '-d');

    set(gca, 'YScale', 'log', 'FontSize', 16,'LineWidth',2);
    grid on; box on;
    xlabel('masked perturbation pixels','FontSize',16);
    ylabel('wrapped phase RMSE vs GT [rad]','FontSize',16);
    % title(sprintf('Perturbation-size sweep, noise %.1e, n = %d random realizations', nl, nRealizations));
    legend(labels, 'Location','best', 'Interpreter','none','FontSize',16);
    axis tight;
    
    if save_results
        tag = sprintf('noise_%0.1e', nl);
        tag = strrep(tag, '+', '');
        tag = strrep(tag, '-', 'm');
        tag = strrep(tag, '.', 'p');
        saveas(gcf, fullfile(out_dir, ['wrapped_phase_rmse_vs_size_' tag '.png']));
        savefig(gcf, fullfile(out_dir, ['wrapped_phase_rmse_vs_size_' tag '.fig']));
    end
end
end


function plot_phase_vs_noise_summary(S, mask_size_px, nRealizations, out_dir, save_results)

% labels = {'dDBFH | cap/string valid region', ...
%     'dOTF | cap/string perturbation', ...
%     'dDBFH | dOTF-opt valid region', ...
%     'dOTF | dOTF-opt perturbation'};
labels = {'dDBFH | string-cut perturbation', ...
    'dOTF | string-cut perturbation', ...
    'dOTF | dOTF perturbation'};

figure('Color','w','Name',sprintf('Wrapped phase RMSE, mask px size %.1e', mask_size_px), ...
    'Position',[100 100 950 620]);
hold on;

plot_iqr_curve(1./S.NoiseLevel, S.DBFH_same_med, S.DBFH_same_q25, S.DBFH_same_q75, '-o');
plot_iqr_curve(1./S.NoiseLevel, S.dOTF_same_med, S.dOTF_same_q25, S.dOTF_same_q75 ,'-s');
% plot_iqr_curve(1./S.NoiseLevel, S.DBFH_opt_med,  S.DBFH_opt_q25,  S.DBFH_opt_q75,  '-^');
plot_iqr_curve(1./S.NoiseLevel, S.dOTF_opt_med,  S.dOTF_opt_q25,  S.dOTF_opt_q75,  '-d');

set(gca, 'YScale', 'log','XScale', 'log', 'FontSize', 16,'LineWidth',2);
grid on; box on;
xlabel('SNR level','FontSize',16);
ylabel('wrapped phase RMSE vs GT [rad]','FontSize',16);
% title(sprintf('Perturbation-size sweep, noise %.1e, mask px size = %d random realizations', mask_size_px, nRealizations));
legend(labels, 'Location','best', 'Interpreter','none','FontSize',16);

if save_results
    tag = sprintf('mask sz px_%0.1e', mask_size_px);
    tag = strrep(tag, '+', '');
    tag = strrep(tag, '-', 'm');
    tag = strrep(tag, '.', 'p');
    saveas(gcf, fullfile(out_dir, ['wrapped_phase_rmse_vs_size_' tag '.png']));
    savefig(gcf, fullfile(out_dir, ['wrapped_phase_rmse_vs_size_' tag '.fig']));
end

end

function plot_iqr_curve(x, med, q25, q75, lineSpec)
%PLOT_IQR_CURVE Median with asymmetric IQR error bars.
x = x(:);
med = med(:);
q25 = q25(:);
q75 = q75(:);

valid = isfinite(x) & isfinite(med) & isfinite(q25) & isfinite(q75);
x = x(valid);
med = max(med(valid), eps);
q25 = max(q25(valid), eps);
q75 = max(q75(valid), eps);

lo = max(med - q25, 0);
hi = max(q75 - med, 0);
errorbar(x, med, lo, hi, lineSpec, 'LineWidth', 2.5, 'CapSize', 5, 'MarkerSize', 8);
end
