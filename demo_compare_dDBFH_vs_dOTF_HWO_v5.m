%% demo_compare_dDBFH_vs_dOTF_HWO_joint_mask.m
% Compare diff-DBFH against dOTF on the same HWO-like segmented aperture.
%
% The final comparison is performed on a joint valid mask:
%   joint_valid = diff-DBFH-valid pixels ∩ dOTF-valid pixels.
% This prevents an unfair comparison between a full-support diff-DBFH estimate
% and a dOTF estimate with missing/contaminated pupil regions.
%
% Two dOTF branches are evaluated:
%   1) a compact edge/corner perturbation with the SAME number of masked
%      pupil pixels as the diff-DBFH cap/string;
%   2) the exact same cap/string perturbation used by diff-DBFH.
%
% Naming convention for image captures:
%   I_full : PSF of the unperturbed full comparison pupil.
%   I_prtb : PSF after the perturbing mask/block is inserted.
%
% For the diff-DBFH branch, I_prtb_dbfh is the capture with the E2 cap blocked,
% and I_full is the capture with E1+E2 present. Therefore:
%   DeltaI = I_full - I_prtb_dbfh
%          = 2*real(A_prtb_dbfh^* A_mask_dbfh) + |A_mask_dbfh|^2.
%
% For the dOTF branches:
%   dOTF = IFFT2c(I_prtb - I_full).
%
% Run from the project root containing:
%   ImZ_from_ReZ/    diff_DBFH/    fftc/
%
% 20 May 2026

clear; close all; clc;

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

rng(1);

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

% Perturbation geometry in pupil pixels.
% With physical sampling enabled, the old value string_depth_pix=10 is a
% much smaller fraction of a segment than it was for seg_radius_pix=40.
% Keep it fixed only if that is intentional.
string_depth_pix  = 25;
gap_pix           = 1;

% Backward-compatible variable names used later in the script.
Q = Q_pix_per_lambdaD_eff;

% Noise model: independent additive Gaussian noise on each PSF image,
% scaled by max(I_full(:)).
noise_level       = 1e-5;
use_noisy_mag1_for_dDBFH = true;

% Cross-term recovery controls for dDBFH.
d_theta           = 0.1;
theta_list        = 0:d_theta:180-d_theta;
use_exact_ReZ     = false;   % false: ReZ = DeltaI/2 approximation
use_true_Z_solver = false;   % true: bypass KK and feed Z_true

% Solver controls copied from the working GPU demo.
useGPU            = true;
nsamp             = 30;
blockSize         = 1;
precond_cap       = 1e2;
lsqr_tol          = 1e-8;
lsqr_maxit        = 1e4;
K_threshold       = 1e-3;
supportScale      = 1;
crossWeight       = 100;
run_adjoint_tests = false;

% dOTF controls.
% copy_choice: 'plus', 'minus', or 'autoByCoverage'.
% The default does NOT use the truth to select the copy.
dotf_opts = struct();
dotf_opts.copy_choice = 'autoByCoverage';
dotf_opts.exclude_block_pixels = true;

show_plots        = true;
save_results      = false;
out_dir           = fullfile(project_root, 'results_dDBFH_vs_dOTF');

if save_results && ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% ========================================================================
%  BUILD SEGMENTED APERTURE AND dDBFH CAP/STRING PERTURBATION
%  ========================================================================

[seg_id, seg_centers, X, Y] = make_hex_segment_aperture(N, seg_radius_pix, num_rings);
aperture = seg_id > 0;
nSeg = max(seg_id(:));

% Same perturbation as demo_differential_DBFH_gpu.m: the upper-most
% peripheral segment gets a thin outer cap E2, with a one-pixel gap.
[~, top_seg] = max(seg_centers(:,2));
top_mask = seg_id == top_seg;

c_top = seg_centers(top_seg,:);
n_out = c_top / norm(c_top);
S = X*n_out(1) + Y*n_out(2);
S_top = S(top_mask);
s_edge = max(S_top);

cut_E2  = s_edge - string_depth_pix;
cut_gap = cut_E2 - gap_pix;

supp2_dbfh   = top_mask & (S >= cut_E2);                         % pixels masked in I_prtb_dbfh
gap_mask = top_mask & (S >= cut_gap) & (S < cut_E2);
supp1_dbfh = aperture & ~supp2_dbfh & ~gap_mask;                 % E1 / perturbed pupil support
supp12_dbfh  = supp1_dbfh | supp2_dbfh;                         % common comparison pupil support

nMask = nnz(supp2_dbfh);
supp2_dotf_same_npix = make_codona_blocker(supp12_dbfh, X, Y, nMask);
supp1_dotf_same_npix = supp12_dbfh - supp2_dotf_same_npix;

fprintf('\n===== Perturbation pixel accounting =====\n');
fprintf('dDBFH cap/string masked pixels  = %d\n', nnz(supp2_dbfh));
fprintf('dOTF optimal masked pixels      = %d\n', nnz(supp2_dotf_same_npix));

plot_supports(supp12_dbfh,supp1_dbfh,supp1_dotf_same_npix);

%% ========================================================================
%  RANDOM SEGMENTED PHASE FRONT
%  ========================================================================
% 
% E = zeros(N,N);
% 
% for k = 1:nSeg
%     mk = seg_id == k;
%     if ~any(mk(:)), continue; end
% 
%     cx = seg_centers(k,1);
%     cy = seg_centers(k,2);
% 
%     dx = (X - cx) / seg_radius_pix;
%     dy = (Y - cy) / seg_radius_pix;
% 
%     piston = 1.0 * randn();
%     tip    = 1.0 * randn();
%     tilt   = 1.0 * randn();
%     defoc  = 1.0 * randn();
% 
%     rho2 = dx.^2 + dy.^2;
%     rho2_mean = mean(rho2(mk));
% 
%     phi = piston + tip*dx + tilt*dy + defoc*(rho2 - rho2_mean);
%     E(mk) = exp(1i * phi(mk));
% end

E = make_random_segmented_field(seg_id, seg_centers, X, Y, seg_radius_pix, nSeg, zernike_nMax, zernike_coeff_sigma_rad);

% dDBFH perturbation fields
E12 = E .* supp12_dbfh;
E1_dbfh = E .* supp1_dbfh;
E2_dbfh = E .* supp2_dbfh;

% dOTF optimal perturbation fields
E1_dotf = E12;
E1_dotf(supp2_dotf_same_npix) = 0;

if 0
    figure;
    ax(1) = subplot(3,1,1);imagesc(real(E12));title('diff DBFH full aperture');
    ax(2) = subplot(3,1,2);imagesc(real(E1_dbfh));title('diff DBFH partial aperture');
    ax(3) = subplot(3,1,3);imagesc(real(E1_dotf));title('dOTF partial aperture');
end
%% ========================================================================
%  FORWARD IMAGES: I12 AND I1
%  ========================================================================

A12 = fft2c(E12);
I12_clean = abs(A12).^2;

A1_dbfh = fft2c(E1_dbfh);
A2_dbfh = fft2c(E2_dbfh); % for debug purposes only
I1_dbfh_clean = abs(A1_dbfh).^2;

A1_dotf = fft2c(E1_dotf);
I1_dotf_clean = abs(A1_dotf).^2;

noise_scale = max(I12_clean(:));
I12     = I12_clean     + noise_level * noise_scale * randn(size(I12_clean));
I1_dbfh = I1_dbfh_clean + noise_level * noise_scale * randn(size(I12_clean));
I1_dotf = I1_dotf_clean + noise_level * noise_scale * randn(size(I12_clean));

%% ========================================================================
%  dDBFH: RECOVER COMPLEX CROSS TERM, THEN SOLVE HARD-ANCHOR SYSTEM
%  ========================================================================

fprintf('\n===== dDBFH: cross-term construction =====\n');

Z_true   = conj(A1_dbfh) .* A2_dbfh;  % solver expects A1^* A2
ReZ_exact = real(Z_true);
ImZ_true  = imag(Z_true);

DeltaI_dbfh = I12 - I1_dbfh; % noisy versions

if use_exact_ReZ
    ReZ_meas = ReZ_exact;
else
    % First-order differential approximation: DeltaI = 2 Re(A1^*A2)+|A2|^2.
    % The |A2|^2 bias is deliberately kept; do not silently subtract it.
    ReZ_meas = 0.5 * DeltaI_dbfh;
end

CSg = correlation_support(supp1_dbfh, supp2_dbfh);

if use_true_Z_solver
    Z_solver = Z_true;
    ImZ_rec = ImZ_true;
    kk = struct();
    solver_label = 'true Z = A_prtb^* A_mask';
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
    solver_label = 'recovered Z = ReZ_meas + i ImZ_rec';
end

rel_ReZ_err = relnorm(ReZ_meas - ReZ_exact, ReZ_exact, true(size(ReZ_exact)));
rel_ImZ_err = relnorm(ImZ_rec  - ImZ_true,  ImZ_true,  true(size(ImZ_true)));
rel_Z_err   = relnorm(Z_solver - Z_true,    Z_true,    true(size(Z_true)));

fprintf('noise_level                  = %.3e\n', noise_level);
fprintf('use_noisy_mag1_for_dDBFH     = %d\n', use_noisy_mag1_for_dDBFH);
fprintf('use_exact_ReZ / true_Z        = %d / %d\n', use_exact_ReZ, use_true_Z_solver);
fprintf('relative ReZ error           = %.3e\n', rel_ReZ_err);
fprintf('relative ImZ error           = %.3e\n', rel_ImZ_err);
fprintf('relative complex Z error     = %.3e\n', rel_Z_err);
if isfield(kk, 'line_rel_err')
    fprintf('KK line-domain Im-channel err = %.3e\n', kk.line_rel_err);
end

if use_noisy_mag1_for_dDBFH
    I_mag_dbfh = I1_dbfh;
else
    I_mag_dbfh = I1_dbfh_clean;
end
mag1 = sqrt(max(I_mag_dbfh, 0));

fprintf('\n===== dDBFH: hard-anchor LSQR solve =====\n');

if useGPU
    assert(gpuDeviceCount > 0, 'useGPU=true but no CUDA GPU is available.');
    gd = gpuDevice;
    fprintf('Using GPU: %s, available memory %.2f GB\n', gd.Name, gd.AvailableMemory/1024^3);

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

fprintf('solver cross term             = %s\n', solver_label);
fprintf('sys.n = nX1_free + nE2        = %d = %d + %d\n', sys.n, sys.nX1, sys.nE2);
fprintf('valid focal K pixels          = %d\n', sys.nK);
fprintf('hard anchor index             = %d\n', sys.anchor_idx);
fprintf('K threshold / crossWeight     = %.1e / %.3g\n', K_threshold, crossWeight);

if run_adjoint_tests
    fprintf('Adjoint tests are not expanded here. Use your existing demo block if needed.\n');
end

fprintf('\n===== dDBFH: Hutchinson right preconditioner =====\n');
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

fprintf('p length/sys.n = %d / %d\n', numel(p), sys.n);
fprintf('p uncapped range = %.3e / %.3e\n', min(to_cpu(p0)), max(to_cpu(p0)));
fprintf('p capped range   = %.3e / %.3e\n', min(to_cpu(p)),  max(to_cpu(p)));

Afun = make_Afun_lsqr_stacked_diagprecond(sys, p);
bstack = [real(sys.b); imag(sys.b)];

fprintf('\n===== dDBFH: LSQR solve =====\n');
tic;
[ustack, dbfh_flag, dbfh_relres, dbfh_iter, dbfh_resvec] = lsqr(Afun, bstack, lsqr_tol, lsqr_maxit);
ddb_time = toc;

ur = ustack(1:sys.n);
ui = ustack(sys.n+1:end);
xvec = p .* (ur + 1i*ui);

if useGPU
    dbfh_unpack = diff_dbfh_unpack_solution_gpu(sys, xvec, 'normalizeX1', false);
    dbfh_unpack = gather_struct(dbfh_unpack);
    sys_small = gather_sys_nohandles(sys);
else
    dbfh_unpack = diff_dbfh_unpack_solution(sys, xvec, 'normalizeX1', false);
    sys_small = gather_sys_nohandles(sys);
end

E_dbfh_raw = dbfh_unpack.E;
[~, E_dbfh_scaled, alpha_dbfh] = align_complex_scalar(E12, E_dbfh_raw, supp12_dbfh);

dbfh = struct();
dbfh.method = 'dDBFH';
dbfh.sys = sys_small;
dbfh.unpack = dbfh_unpack;
dbfh.E_rec_raw = E_dbfh_raw;
dbfh.E_rec_scaled = E_dbfh_scaled;
dbfh.alpha = alpha_dbfh;
dbfh.flag = dbfh_flag;
dbfh.relres = double(to_cpu(dbfh_relres));
dbfh.iter = dbfh_iter;
dbfh.resvec = to_cpu(dbfh_resvec);
dbfh.solve_time = ddb_time;
dbfh.leak1 = dbfh_unpack.leak1;
dbfh.leak2 = dbfh_unpack.leak2;
dbfh.cross_relres = dbfh_unpack.cross_relres;

fprintf('dDBFH flag/relres/iter/time  = %d / %.3e / %d / %.2f s\n', ...
    dbfh.flag, dbfh.relres, dbfh.iter, dbfh.solve_time);
fprintf('dDBFH full phase RMSE/scalar field relerr = %.3e rad / %.3e\n', ...
    phase_rmse(E12, dbfh.E_rec_scaled, supp12_dbfh), ...
    relnorm(dbfh.E_rec_scaled - E12, E12, supp12_dbfh));

%% ========================================================================
%  dOTF BRANCHES: COMPACT EDGE/CORNER AND SAME CAP/STRING PERTURBATION
%  ========================================================================

% For the paper comparison, keep the dOTF-favorable edge perturbation with
% the same number of blocked pupil pixels as the diff-DBFH cap/string.
fprintf('\n===== dOTF: compact edge/corner perturbation, same masked-pixel count =====\n');
dotf_edge = run_dOTF_blocked_probe(I12, I1_dotf, E12, supp12_dbfh, supp2_dotf_same_npix, X, Y, dotf_opts);

% Also evaluate dOTF with the exact same cap/string perturbation used by
% diff-DBFH. This isolates method behavior from perturbation-geometry choice.
fprintf('\n===== dOTF: same cap/string perturbation as diff-DBFH =====\n');
dotf_same = run_dOTF_blocked_probe(I12, I1_dbfh, E12, supp12_dbfh, supp2_dbfh, X, Y, dotf_opts);

%% ========================================================================
%  COMMON-MASK METRICS
%  ========================================================================

% Define one joint comparison mask across all plotted methods. This removes
% pixels that are invalid for any method. diff-DBFH is defined over most of
% the comparison support, while dOTF has missing/contaminated regions from
% copy overlap and the perturbation mini-OTF. The intersection prevents an
% unfair full-support vs valid-only comparison.
dbfh_valid = supp1_dbfh & abs(E12) > 0 & ...
    isfinite(real(dbfh.E_rec_raw)) & isfinite(imag(dbfh.E_rec_raw));

dotf_edge_valid = dotf_edge.valid & abs(E12) > 0 & ...
    isfinite(real(dotf_edge.raw)) & isfinite(imag(dotf_edge.raw));

dotf_same_valid = dotf_same.valid & abs(E12) > 0 & ...
    isfinite(real(dotf_same.raw)) & isfinite(imag(dotf_same.raw));

joint_valid = dbfh_valid & dotf_edge_valid & dotf_same_valid;

if nnz(joint_valid) == 0
    error('Joint valid mask is empty. Check dOTF extraction/support geometry.');
end

% Re-align every reconstruction on the SAME joint mask. Do not reuse scalars
% fitted on different regions, otherwise the comparison is subtly biased.
[~, E_dbfh_joint, alpha_dbfh_joint] = align_complex_scalar(E12, dbfh.E_rec_raw, joint_valid);
[~, E_dotf_edge_joint, alpha_dotf_edge_joint] = align_complex_scalar(E12, dotf_edge.raw, joint_valid);
[~, E_dotf_same_joint, alpha_dotf_same_joint] = align_complex_scalar(E12, dotf_same.raw, joint_valid);

m_dbfh_joint        = metric_pack(E12, E_dbfh_joint,        joint_valid, dbfh.solve_time, dbfh.iter, dbfh.relres);
m_dotf_edge_joint = metric_pack(E12, E_dotf_edge_joint, joint_valid, dotf_edge.solve_time, NaN, NaN);
m_dotf_same_joint    = metric_pack(E12, E_dotf_same_joint,    joint_valid, dotf_same.solve_time, NaN, NaN);

phase_err_dbfh         = angle(conj(E12(joint_valid)) .* E_dbfh_joint(joint_valid));
phase_err_dotf_edge = angle(conj(E12(joint_valid)) .* E_dotf_edge_joint(joint_valid));
phase_err_dotf_same    = angle(conj(E12(joint_valid)) .* E_dotf_same_joint(joint_valid));

abs_phase_err_dbfh         = abs(phase_err_dbfh);
abs_phase_err_dotf_edge = abs(phase_err_dotf_edge);
abs_phase_err_dotf_same    = abs(phase_err_dotf_same);

Method = {'diff-DBFH'; 'dOTF edge/corner'; 'dOTF same cap/string'};
Mask = {'joint valid mask'; 'joint valid mask'; 'joint valid mask'};
MaskedPixels = [nMask; nMask; nMask];
JointCoveragePct = 100 * [nnz(joint_valid); nnz(joint_valid); nnz(joint_valid)] / nnz(supp12_dbfh);
PhaseRMSE_rad = [m_dbfh_joint.phase_rmse; m_dotf_edge_joint.phase_rmse; m_dotf_same_joint.phase_rmse];
PhaseMAE_rad = [mean(abs_phase_err_dbfh); mean(abs_phase_err_dotf_edge); mean(abs_phase_err_dotf_same)];
PhaseMedianAbs_rad = [median(abs_phase_err_dbfh); median(abs_phase_err_dotf_edge); median(abs_phase_err_dotf_same)];
FieldRelErr_scalar = [m_dbfh_joint.field_relerr; m_dotf_edge_joint.field_relerr; m_dotf_same_joint.field_relerr];
AmpRelErr_scalar = [m_dbfh_joint.amp_relerr; m_dotf_edge_joint.amp_relerr; m_dotf_same_joint.amp_relerr];
Time_s = [m_dbfh_joint.time_s; m_dotf_edge_joint.time_s; m_dotf_same_joint.time_s];
LSQRIter = [m_dbfh_joint.iter; m_dotf_edge_joint.iter; m_dotf_same_joint.iter];
LSQRRelres = [m_dbfh_joint.relres; m_dotf_edge_joint.relres; m_dotf_same_joint.relres];

metrics_joint = table(Method, Mask, MaskedPixels, JointCoveragePct, ...
    PhaseRMSE_rad, PhaseMAE_rad, PhaseMedianAbs_rad, ...
    FieldRelErr_scalar, AmpRelErr_scalar, Time_s, LSQRIter, LSQRRelres);

fprintf('\n===== Joint-mask comparison =====\n');
fprintf('Joint valid pixels            = %d\n', nnz(joint_valid));
fprintf('Joint coverage                = %.2f %% of supp12_dbfh\n', 100*nnz(joint_valid)/nnz(supp12_dbfh));
disp(metrics_joint);

results = struct();
results.metrics_joint = metrics_joint;
results.ddb = dbfh;
results.dotf_edge = dotf_edge;
results.dotf_same = dotf_same;
results.params = struct('N',N,'seg_radius_pix',seg_radius_pix, ...
    'seg_radius_pix_exact',seg_radius_pix_exact, ...
    'num_rings',num_rings, ...
    'D_m_inscribed',D_m_inscribed,'f_m',f_m,'lambda_m',lambda_m,'cam_px_m',cam_px_m, ...
    'L_m_pupil',L_m_pupil,'px_m_pupil_plane',px_m_pupil_plane, ...
    'D_pix_inscribed',D_pix_inscribed, ...
    'D_pix_inscribed_eff',D_pix_inscribed_eff, ...
    'px_m_pupil_plane_eff',px_m_pupil_plane_eff, ...
    'L_m_pupil_eff',L_m_pupil_eff, ...
    'px_m_focal_plane_eff',px_m_focal_plane_eff, ...
    'Q_pix_per_lambdaD_eff',Q_pix_per_lambdaD_eff, ...
    'lambdaD_per_pix_eff',lambdaD_per_pix_eff, ...
    'string_depth_pix',string_depth_pix,'gap_pix',gap_pix,'noise_level',noise_level, ...
    'nMask',nMask,'use_noisy_mag1_for_dDBFH',use_noisy_mag1_for_dDBFH, ...
    'use_exact_ReZ',use_exact_ReZ,'use_true_Z_solver',use_true_Z_solver, ...
    'K_threshold',K_threshold,'crossWeight',crossWeight);
results.truth = struct('E12',E12,'E_prtb_dbfh',E1_dbfh, ...
    'E1_dotf',E1_dotf,'E2_dbfh',E2_dbfh, ...
    'supp1_dbfh',supp1_dbfh,'supp2_dbfh',supp2_dbfh,'supp12_dbfh',supp12_dbfh, ...
    'supp2_dotf_edge',supp2_dotf_same_npix, ...
    'gap_mask',gap_mask,'seg_id',seg_id,'seg_centers',seg_centers);
results.cross_term = struct('Z_true',Z_true,'Z_solver',Z_solver,'ReZ_meas',ReZ_meas, ...
    'ImZ_rec',ImZ_rec,'rel_ReZ_err',rel_ReZ_err,'rel_ImZ_err',rel_ImZ_err, ...
    'rel_Z_err',rel_Z_err,'kk',kk);
results.joint = struct('valid',joint_valid, ...
    'dbfh_valid',dbfh_valid, ...
    'dotf_edge_valid',dotf_edge_valid, ...
    'dotf_same_valid',dotf_same_valid, ...
    'E_dbfh_joint',E_dbfh_joint, ...
    'E_dotf_edge_joint',E_dotf_edge_joint, ...
    'E_dotf_same_joint',E_dotf_same_joint, ...
    'alpha_dbfh_joint',alpha_dbfh_joint, ...
    'alpha_dotf_edge_joint',alpha_dotf_edge_joint, ...
    'alpha_dotf_same_joint',alpha_dotf_same_joint, ...
    'phase_err_dbfh',phase_err_dbfh, ...
    'phase_err_dotf_edge',phase_err_dotf_edge, ...
    'phase_err_dotf_same',phase_err_dotf_same);

if save_results
    save(fullfile(out_dir, 'dDBFH_vs_dOTF_HWO_joint_mask_threeway_results.mat'), 'results', '-v7.3');
    writetable(metrics_joint, fullfile(out_dir, 'dDBFH_vs_dOTF_HWO_joint_mask_threeway_metrics.csv'));
end

%% ========================================================================
%  PLOTS
%  ========================================================================

if show_plots
    plot_joint_mask_comparison(E12, E_dbfh_joint, E_dotf_edge_joint, E_dotf_same_joint, joint_valid, metrics_joint);
end

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function dotf = run_dOTF_blocked_probe(I_full, I_prtb, E12, supp12_dbfh, block_mask, X, Y, opts)
%RUN_DOTF_BLOCKED_PROBE Codona-style two-image dOTF extraction.
%
% I_full     : PSF of the unperturbed full pupil E12.
% I_prtb     : PSF after inserting the perturbing opaque block/mask.
% E12     : complex truth field on the unperturbed pupil, for metrics only.
% supp12_dbfh  : support of E12.
% block_mask : pupil pixels blocked in the perturbed capture.
%
% The reconstruction is extracted from dOTF = IFFT2c(I_prtb - I_full).
% The unknown scalar from the local perturbation is removed only for metric
% calculation by a complex least-squares scalar alignment.

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

fprintf('Computing dOTF = IFFT2c(I_prtb - I_full)...\n');
tic;
dO = ifft2c(I_prtb - I_full);

x0 = round(mean(X(block_mask)));
y0 = round(mean(Y(block_mask)));

CS_full_block  = correlation_support(supp12_dbfh, block_mask);  % support of IFT[A_full^* A_block]
CS_block_full  = correlation_support(block_mask, supp12_dbfh);  % support of IFT[A_block^* A_full]
CS_block_block = correlation_support(block_mask, block_mask); % mini-OTF of perturbation

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

fprintf('selected copy                  = %s\n', dotf.copy_name);
fprintf('block centroid [x0,y0]          = [%d,%d]\n', dotf.x0, dotf.y0);
fprintf('masked pixels                  = %d\n', nnz(block_mask));
fprintf('valid coverage                 = %.2f %% of supp12_dbfh\n', 100*nnz(dotf.valid)/nnz(supp12_dbfh));
fprintf('phase RMSE / scalar relerr      = %.3e rad / %.3e\n', dotf.phase_rmse, dotf.field_relerr);
end

function cand = extract_dOTF_blocked_copy(dO, E12, supp12_dbfh, block_mask, X, Y, ...
    x0, y0, CS_full_block, CS_block_full, CS_block_block, mode, exclude_block_pixels)
%EXTRACT_DOTF_BLOCKED_COPY Extract one of the two shifted dOTF pupil copies.

[Nr,Nc] = size(dO);
cx = round((Nc+1)/2);
cy = round((Nr+1)/2);

switch lower(mode)
    case 'plus'
        % IFT[A_full^* A_block] is sampled at s = x_block - x_pupil.
        % It contains conj(E12) times the local block field; conjugate it.
        sx = x0 - X;
        sy = y0 - Y;
        copy_name = 'plus: conj IFT[A_full^* A_block], s=x0-x';
        use_conj = true;
        ownCS = CS_full_block;
        otherCS = CS_block_full;
    case 'minus'
        % IFT[A_block^* A_full] is sampled at s = x_pupil - x_block.
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
    field_relerr_val = relnorm(E_rec_scaled - E12, E12, valid);
    amp_relerr_val = relnorm(abs(E_rec_scaled) - abs(E12), abs(E12), valid);
else
    E_rec_scaled = raw;
    alpha = NaN;
    phase_rmse_val = Inf;
    field_relerr_val = Inf;
    amp_relerr_val = Inf;
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
cand.field_relerr = field_relerr_val;
cand.amp_relerr = amp_relerr_val;
end

function B = make_codona_blocker(supp12_dbfh, X, Y, nMask)
% make_codona_blocker dOTF-favorable block near an outer corner.
% Chooses nMask pixels in supp12_dbfh closest to the most radially outward
% pupil pixel. This approximates a edge/corner bite and preserves
% exactly the same number of masked pixels as the diff-DBFH perturbation.

idx = find(supp12_dbfh);

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

function m = metric_pack(Ftrue, Frec, mask, time_s, iter, relres)
mask = logical(mask) & isfinite(real(Frec)) & isfinite(imag(Frec)) & abs(Ftrue) > 0;
m = struct();
m.npix = nnz(mask);
m.phase_rmse = phase_rmse(Ftrue, Frec, mask);
m.field_relerr = relnorm(Frec - Ftrue, Ftrue, mask);
m.amp_relerr = relnorm(abs(Frec) - abs(Ftrue), abs(Ftrue), mask);
m.time_s = time_s;
m.iter = iter;
m.relres = relres;
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

function S2 = gather_sys_nohandles(S)
% Keep scalar/mask diagnostics but avoid saving operator handles.
S2 = struct();
keep = {'m','n','nX1','nE2','nK','K','Kfree','crossMask','idxCross','idx1_out', ...
    'idxE2','H','W','anchor_idx','anchorVal','hardAnchor','mag1Floor','cFloor', ...
    'lambdaX1','lambdaE2','supportScale','crossWeight','fftConvention','description'};
for k = 1:numel(keep)
    if isfield(S, keep{k})
        v = S.(keep{k});
        if isa(v, 'gpuArray'), v = gather(v); end
        S2.(keep{k}) = v;
    end
end
end

function C = strip_heavy_candidate(C)
% Keep candidate diagnostics without duplicating full raw/reconstruction arrays.
if isfield(C, 'raw'), C = rmfield(C, 'raw'); end
if isfield(C, 'E_rec_scaled'), C = rmfield(C, 'E_rec_scaled'); end
end

function addpath_if_dir(p)
if exist(p, 'dir')
    addpath(p);
end
end


function plot_joint_mask_comparison(Ftrue, Fdbfh, Fdotf_edge, Fdotf_same, joint_valid, metrics_joint)
%PLOT_JOINT_MASK_COMPARISON One fair figure comparing diff-DBFH and dOTF.
%
% All maps and histograms use the same joint_valid mask. Here the mask is
% the intersection of the valid regions for diff-DBFH, edge-perturbation
% dOTF, and same-cap/string-perturbation dOTF.

joint_valid = logical(joint_valid) & abs(Ftrue) > 0 & ...
    isfinite(real(Fdbfh)) & isfinite(imag(Fdbfh)) & ...
    isfinite(real(Fdotf_edge)) & isfinite(imag(Fdotf_edge)) & ...
    isfinite(real(Fdotf_same)) & isfinite(imag(Fdotf_same));

if nnz(joint_valid) == 0
    error('plot_joint_mask_comparison received an empty joint_valid mask.');
end

N = size(Ftrue,1);

ph_true = angle(Ftrue);
ph_dbfh = angle(Fdbfh);
ph_dotf_edge = angle(Fdotf_edge);
ph_dotf_same = angle(Fdotf_same);

err_dbfh         = angle(conj(Ftrue) .* Fdbfh);
err_dotf_edge = angle(conj(Ftrue) .* Fdotf_edge);
err_dotf_same    = angle(conj(Ftrue) .* Fdotf_same);

e_dbfh         = err_dbfh(joint_valid);
e_dotf_edge = err_dotf_edge(joint_valid);
e_dotf_same    = err_dotf_same(joint_valid);
e_all = [e_dbfh(:); e_dotf_edge(:); e_dotf_same(:)];
e_all = e_all(isfinite(e_all));

if isempty(e_all)
    hist_lim = pi;
else
    s = sort(abs(e_all));
    hist_lim = s(max(1, round(0.995*numel(s))));
    hist_lim = min(pi, max(1e-6, 1.05*hist_lim));
end
edges = linspace(-hist_lim, hist_lim, 101);

rmse_dbfh         = metrics_joint.PhaseRMSE_rad(1);
rmse_dotf_edge = metrics_joint.PhaseRMSE_rad(2);
rmse_dotf_same    = metrics_joint.PhaseRMSE_rad(3);

figure('Color','w','Name','diff-DBFH vs dOTF: joint-mask comparison', ...
    'Position',[60 60 1850 950]);
tiledlayout(2,4,'TileSpacing','tight','Padding','tight');

ax = gobjects(2,4);

ax(1) = nexttile;
imagesc_masked(ph_true, joint_valid);
title('ground truth phase on joint mask');
clim([-pi pi]);
hold on; text(0.05,0.9,'(a)','fontsize',24,'Units','normalized','color','k');
set(gca,'LineWidth',2);

ax(2) = nexttile;
imagesc_masked(ph_dbfh, joint_valid);
title('diff-DBFH phase (string-cut pert.)');
clim([-pi pi]);
hold on; text(0.05,0.9,'(b)','fontsize',24,'Units','normalized','color','k');
set(gca,'LineWidth',2);

ax(3) = nexttile;
imagesc_masked(ph_dotf_edge, joint_valid);
title('dOTF phase (edge pert.)');
clim([-pi pi]);
hold on; text(0.05,0.9,'(c)','fontsize',24,'Units','normalized','color','k');
set(gca,'LineWidth',2);

ax(4) = nexttile;
imagesc_masked(ph_dotf_same, joint_valid);
title('dOTF phase (string-cut pert.)');
clim([-pi pi]);
hold on; text(0.05,0.9,'(d)','fontsize',24,'Units','normalized','color','k');
set(gca,'LineWidth',2);

nexttile;
histogram(e_dbfh, edges, 'Normalization','probability', 'DisplayStyle','stairs', 'LineWidth',2);
hold on;
histogram(e_dotf_edge, edges, 'Normalization','probability', 'DisplayStyle','stairs', 'LineWidth',2);
histogram(e_dotf_same, edges, 'Normalization','probability', 'DisplayStyle','stairs', 'LineWidth',2);
grid on;
xlabel('phase error [rad]');
ylabel('probability');
legend('diff-DBFH','dOTF (edge pert.)','dOTF (string-cut pert.)','Location','northoutside');
% title('phase-error histogram on joint mask');
hold on; text(0.05,0.9,'(e)','fontsize',24,'Units','normalized','color','k');
set(gca,'LineWidth',2,'FontSize',16);

ax(6) = nexttile;
imagesc_masked(err_dbfh, joint_valid);
title(sprintf('diff-DBFH error (string-cut pert.), RMSE %.3g rad', rmse_dbfh));
clim([-hist_lim hist_lim]);
hold on; text(0.05,0.9,'(f)','fontsize',24,'Units','normalized','color','k');
set(gca,'LineWidth',2);

ax(7) = nexttile;
imagesc_masked(err_dotf_edge, joint_valid);
title(sprintf('dOTF error (edge pert.), RMSE %.3g rad', rmse_dotf_edge));
clim([-hist_lim hist_lim]);
hold on; text(0.05,0.9,'(g)','fontsize',24,'Units','normalized','color','k');
set(gca,'LineWidth',2);

ax(8) = nexttile;
imagesc_masked(err_dotf_same, joint_valid);
title(sprintf('dOTF error (string-cut pert.), RMSE %.3g rad', rmse_dotf_same));
clim([-hist_lim hist_lim]);
hold on; text(0.05,0.9,'(h)','fontsize',24,'Units','normalized','color','k');
set(gca,'LineWidth',2);

colormap parula;
linkaxes(ax);
xlim(N/2+N/6*[-1,1]);
ylim(N/2+N/6*[-1,1]);

end


function plot_supports(supp_full, supp_diff_DBFH_pert, supp_dOTF_pert)

custom_colormap = [0,0,0;1,1,1;0.5,0.5,0.5];

N = size(supp_full,1);
clims = [-1,1];
pert_support_diff_DBFH = supp_full - supp_diff_DBFH_pert;
pert_support_dOTF      = supp_full - supp_dOTF_pert;

figure('Color','w','Name','Supports','Position',[100 100 1000 650]);

ax(1) = subplot(2,3,1); 
imagesc(supp_full,clims); axis image; title('full aperture');xticks([]);yticks([]);
hold on; text(0.05,0.9,'(a)','fontsize',24,'Units','normalized','color','k');

ax(2) = subplot(2,3,2);
imagesc(double(supp_diff_DBFH_pert) - double(pert_support_diff_DBFH),clims); axis image; title('diff-DBFH string-cut perturbation');xticks([]);yticks([]);
hold on; text(0.05,0.9,'(b)','fontsize',24,'Units','normalized','color','k');

ax(3) = subplot(2,3,3);
imagesc(double(supp_dOTF_pert) - double(pert_support_dOTF),clims); axis image; title('dOTF edge perturbation');xticks([]);yticks([]);
hold on; text(0.05,0.9,'(c)','fontsize',24,'Units','normalized','color','k');

ax(4) = subplot(2,3,4);
imagesc(correlation_support(supp_diff_DBFH_pert, pert_support_diff_DBFH),clims); axis image; title({'F^{-1}[A1^*A2] support','in string-cut perturbation'});
hold on;plot(N/2+1,N/2+1,'ok','MarkerFaceColor','k','MarkerSize',4);xticks([]);yticks([]);
text(0.05,0.9,'(d)','fontsize',24,'Units','normalized','color','k');
plot([0;N],N/2*[1;1],'--k','LineWidth',1.5);hold off;

ax(5) = subplot(2,3,5);
imagesc(correlation_support(supp_dOTF_pert, pert_support_dOTF),clims); axis image; title({'F^{-1}[A1^*A2] support','in edge perturbation'});
hold on;plot(N/2+1,N/2+1,'ok','MarkerFaceColor','k','MarkerSize',4);xticks([]);yticks([]);
text(0.05,0.9,'(e)','fontsize',24,'Units','normalized','color','k');
theta_line = 35;
y = @(x) tand(theta_line)*(x-N/2) + N/2;
plot([0,N],[y(0) y(N)],'--k','LineWidth',1.5);hold off;

ax(1).Position(3:4) = ax(1).Position(3:4)*1.15;
ax(2).Position(3:4) = ax(2).Position(3:4)*1.15;
ax(3).Position(3:4) = ax(3).Position(3:4)*1.15;
ax(4).Position(3:4) = ax(4).Position(3:4)*1.15;
ax(5).Position(3:4) = ax(5).Position(3:4)*1.15;

ax(5).Position(1) = ax(5).Position(1) + 0.13;
ax(4).Position(1) = ax(4).Position(1) + 0.13;

colormap(custom_colormap);
linkaxes(ax);
xlim([N/6,5/6*N]);
ylim([N/6,5/6*N]);

end

function plot_field_comparison(Ftrue, Frec, mask, figTitle)
mask = logical(mask) & abs(Ftrue) > 0 & isfinite(real(Frec)) & isfinite(imag(Frec));
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
ax(4) = nexttile; imagesc_masked(ph_true, mask);  title('phase truth'); clim([-pi pi]);
ax(5) = nexttile; imagesc_masked(ph_rec, mask);   title('phase reconstruction'); clim([-pi pi]);
ax(6) = nexttile; imagesc_masked(ph_err, mask);   title('wrapped phase error'); clim([-pi pi]);
linkaxes(ax);
sgtitle(figTitle, 'Interpreter','none');
colormap parula;
end

function plot_dotf_products(dotf, figTitle)
figure('Color','w','Name',figTitle,'Position',[100 100 1650 850]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
nexttile; imagesc(log10(abs(dotf.dOTF)+1e-18)); axis image off; colorbar; title('log10 |dOTF|');
nexttile; imagesc(angle(dotf.dOTF)); axis image off; colorbar; clim([-pi pi]); title('arg dOTF');
nexttile; imagesc(dotf.valid); axis image off; colorbar; title('selected valid mask');
nexttile; imagesc(log10(abs(dotf.raw)+1e-18)); axis image off; colorbar; title('log10 |raw extracted copy|');
nexttile; imagesc(angle(dotf.raw)); axis image off; colorbar; clim([-pi pi]); title('arg raw extracted copy');
nexttile; imagesc(dotf.valid_otherCS | dotf.valid_miniCS); axis image off; colorbar; title('excluded: overlap or mini-OTF');
colormap parula;
end

function imagesc_masked(M, mask)
D = M;
D(~mask) = NaN;
imagesc(D, 'AlphaData', ~isnan(D));
set(gca,'Color',[1 1 1]);
axis image off;
cbar = colorbar('fontsize',16);
cbar.Label.String = '[rad]';
cbar.Label.FontSize = 16;
end

function clim_symmetric(ax, vals)
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