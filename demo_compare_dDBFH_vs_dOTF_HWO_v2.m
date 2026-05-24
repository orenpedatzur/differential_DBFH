%% demo_compare_dDBFH_vs_dOTF_HWO_v2.m
% Compare dDBFH against two two-capture dOTF baselines on the same
% HWO-like segmented aperture.
%
% Three method cases are evaluated:
%   1) dDBFH using the original long peripheral cap/string perturbation.
%   2) dOTF using that exact same cap/string perturbation.
%   3) dOTF using a compact edge/corner perturbation, with the SAME number
%      of masked pupil pixels as the dDBFH cap/string.
%
% Naming convention for image captures:
%   I_full : PSF of the unperturbed full comparison pupil.
%   I_prtb : PSF after the perturbing mask/block is inserted.
%
% For the dDBFH branch, I_prtb_dbfh is the capture with the E2 cap blocked,
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

% HWO-like segmented aperture, matching demo_differential_DBFH_gpu.m.
N                 = 2048/2;
seg_radius_pix    = 40/2;
num_rings         = 3;
string_depth_pix  = 10/2;
gap_pix           = 1;

% Noise model: independent additive Gaussian noise on each PSF image,
% scaled by max(I_full(:)).
noise_level       = 1e-4;
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
lsqr_maxit        = 1e3;
K_threshold       = 1e-2;
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
save_results      = true;
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

fprintf('\n===== Perturbation pixel accounting =====\n');
fprintf('dDBFH cap/string masked pixels  = %d\n', nnz(supp2_dbfh));
fprintf('dOTF optimal masked pixels      = %d\n', nnz(supp2_dotf_same_npix));

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
%  dOTF BRANCHES: SAME PERTURBATION AND OPTIMAL COMPACT PERTURBATION
%  ========================================================================

fprintf('\n===== dOTF: same dDBFH cap/string perturbation =====\n');
dotf_same = run_dOTF_blocked_probe(I12, I1_dbfh, E12, supp12_dbfh, supp2_dbfh, X, Y, dotf_opts);

fprintf('\n===== dOTF: compact edge/corner perturbation, same masked-pixel count =====\n');
dotf_opt = run_dOTF_blocked_probe(I12, I1_dotf, E12, supp12_dbfh, supp2_dotf_same_npix, X, Y, dotf_opts);

%% ========================================================================
%  SIDE-BY-SIDE METRICS
%  ========================================================================

m_ddb_full     = metric_pack(E12, dbfh.E_rec_scaled, supp12_dbfh, dbfh.solve_time, dbfh.iter, dbfh.relres);
m_ddb_sameMask = metric_pack(E12, dbfh.E_rec_scaled, dotf_same.valid, dbfh.solve_time, dbfh.iter, dbfh.relres);
m_dotf_same    = metric_pack(E12, dotf_same.E_rec_scaled, dotf_same.valid, dotf_same.solve_time, NaN, NaN);
m_ddb_optMask  = metric_pack(E12, dbfh.E_rec_scaled, dotf_opt.valid, dbfh.solve_time, dbfh.iter, dbfh.relres);
m_dotf_opt     = metric_pack(E12, dotf_opt.E_rec_scaled, dotf_opt.valid, dotf_opt.solve_time, NaN, NaN);

Group = { ...
    'dDBFH full support'; ...
    'same dOTF-valid mask'; ...
    'same dOTF-valid mask'; ...
    'optimal dOTF-valid mask'; ...
    'optimal dOTF-valid mask'};
Method = { ...
    'dDBFH'; ...
    'dDBFH'; ...
    'dOTF same cap/string'; ...
    'dDBFH'; ...
    'dOTF compact edge/corner'};
Perturbation = { ...
    'dDBFH cap/string'; ...
    'dDBFH cap/string'; ...
    'dDBFH cap/string'; ...
    'dDBFH cap/string reconstruction evaluated on dOTF-opt mask'; ...
    'compact edge/corner'};
MaskedPixels = [nMask; nMask; nMask; nMask; nMask];
CoveragePct = 100 * [ ...
    nnz(supp12_dbfh); ...
    nnz(dotf_same.valid); ...
    nnz(dotf_same.valid); ...
    nnz(dotf_opt.valid); ...
    nnz(dotf_opt.valid)] / nnz(supp12_dbfh);
PhaseRMSE_rad = [ ...
    m_ddb_full.phase_rmse; ...
    m_ddb_sameMask.phase_rmse; ...
    m_dotf_same.phase_rmse; ...
    m_ddb_optMask.phase_rmse; ...
    m_dotf_opt.phase_rmse];
FieldRelErr_scalar = [ ...
    m_ddb_full.field_relerr; ...
    m_ddb_sameMask.field_relerr; ...
    m_dotf_same.field_relerr; ...
    m_ddb_optMask.field_relerr; ...
    m_dotf_opt.field_relerr];
AmpRelErr_scalar = [ ...
    m_ddb_full.amp_relerr; ...
    m_ddb_sameMask.amp_relerr; ...
    m_dotf_same.amp_relerr; ...
    m_ddb_optMask.amp_relerr; ...
    m_dotf_opt.amp_relerr];
Time_s = [ ...
    m_ddb_full.time_s; ...
    m_ddb_sameMask.time_s; ...
    m_dotf_same.time_s; ...
    m_ddb_optMask.time_s; ...
    m_dotf_opt.time_s];
LSQRIter = [ ...
    m_ddb_full.iter; ...
    m_ddb_sameMask.iter; ...
    m_dotf_same.iter; ...
    m_ddb_optMask.iter; ...
    m_dotf_opt.iter];
LSQRRelres = [ ...
    m_ddb_full.relres; ...
    m_ddb_sameMask.relres; ...
    m_dotf_same.relres; ...
    m_ddb_optMask.relres; ...
    m_dotf_opt.relres];

metrics = table(Group, Method, Perturbation, MaskedPixels, CoveragePct, ...
    PhaseRMSE_rad, FieldRelErr_scalar, AmpRelErr_scalar, Time_s, LSQRIter, LSQRRelres);

disp(metrics);

fprintf('\nInterpretation:\n');
fprintf('  The direct same-data comparison is: dDBFH on same dOTF-valid mask vs dOTF same cap/string.\n');
fprintf('  The method-level two-capture comparison is: dDBFH full/support metrics vs dOTF compact edge/corner,\n');
fprintf('  but coverage differs because single-probe dOTF has a masked overlap region.\n');
fprintf('  Do not compare dDBFH full-pupil RMSE directly against dOTF valid-only RMSE and call that fair.\n');

results = struct();
results.metrics = metrics;
results.ddb = dbfh;
results.dotf_same = dotf_same;
results.dotf_opt = dotf_opt;
results.params = struct('N',N,'seg_radius_pix',seg_radius_pix,'num_rings',num_rings, ...
    'string_depth_pix',string_depth_pix,'gap_pix',gap_pix,'noise_level',noise_level, ...
    'nMask',nMask,'use_noisy_mag1_for_dDBFH',use_noisy_mag1_for_dDBFH, ...
    'use_exact_ReZ',use_exact_ReZ,'use_true_Z_solver',use_true_Z_solver, ...
    'K_threshold',K_threshold,'crossWeight',crossWeight);
results.truth = struct('E12',E12,'E_prtb_dbfh',E1_dbfh, ...
    'E1_dotf',E1_dotf,'E2_dbfh',E2_dbfh, ...
    'supp1_dbfh',supp1_dbfh,'supp2_dbfh',supp2_dbfh,'supp12_dbfh',supp12_dbfh, ...
    'gap_mask',gap_mask,'seg_id',seg_id,'seg_centers',seg_centers);
results.cross_term = struct('Z_true',Z_true,'Z_solver',Z_solver,'ReZ_meas',ReZ_meas, ...
    'ImZ_rec',ImZ_rec,'rel_ReZ_err',rel_ReZ_err,'rel_ImZ_err',rel_ImZ_err, ...
    'rel_Z_err',rel_Z_err,'kk',kk);

if save_results
    save(fullfile(out_dir, 'dDBFH_vs_dOTF_HWO_v2_results.mat'), 'results', '-v7.3');
    writetable(metrics, fullfile(out_dir, 'dDBFH_vs_dOTF_HWO_v2_metrics.csv'));
end

%% ========================================================================
%  PLOTS
%  ========================================================================

if show_plots
    plot_supports(supp12_dbfh, supp2_dbfh, supp2_dotf_same_npix, gap_mask, dotf_same.valid, dotf_opt.valid, CSg);
    plot_field_comparison(E12, dbfh.E_rec_scaled, supp12_dbfh, 'dDBFH reconstruction, full comparison support');
    plot_field_comparison(E12, dbfh.E_rec_scaled, dotf_same.valid, 'dDBFH evaluated on same-perturbation dOTF-valid mask');
    plot_field_comparison(E12, dotf_same.E_rec_scaled, dotf_same.valid, 'dOTF using same cap/string perturbation');
    plot_field_comparison(E12, dbfh.E_rec_scaled, dotf_opt.valid, 'dDBFH evaluated on optimal-dOTF valid mask');
    plot_field_comparison(E12, dotf_opt.E_rec_scaled, dotf_opt.valid, 'dOTF using compact edge/corner perturbation');
    plot_dotf_products(dotf_same, 'dOTF same cap/string products');
    plot_dotf_products(dotf_opt, 'dOTF compact edge/corner products');
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
% pupil pixel. This approximates a compact edge/corner bite and preserves
% exactly the same number of masked pixels as the dDBFH perturbation.

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

function plot_supports(supp12_dbfh, B_dbfh, B_dotf_opt, fixed_gap, valid_same, valid_opt, CSg)
figure('Color','w','Name','Supports','Position',[100 100 1700 650]);
tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
nexttile; imagesc(supp12_dbfh); axis image off; colorbar; title('full comparison support');
nexttile; imagesc(B_dbfh); axis image off; colorbar; title('dDBFH cap/string block');
nexttile; imagesc(B_dotf_opt); axis image off; colorbar; title('dOTF compact edge block');
nexttile; imagesc(fixed_gap); axis image off; colorbar; title('fixed common gap');
nexttile; imagesc(valid_same); axis image off; colorbar; title('dOTF valid: same cap/string');
nexttile; imagesc(valid_opt); axis image off; colorbar; title('dOTF valid: compact edge');
nexttile; imagesc(CSg); axis image off; colorbar; title('CSg for dDBFH Z');
nexttile; imagesc(B_dbfh | B_dotf_opt); axis image off; colorbar; title('both perturbation masks');
colormap parula;
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
colorbar;
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
