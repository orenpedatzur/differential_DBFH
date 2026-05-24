%% demo_compare_dDBFH_vs_dOTF_HWO.m
% Compare differential DBFH against Codona-style dOTF on the same segmented
% HWO-like aperture geometry used in demo_differential_DBFH_gpu.m.
%
% Core comparison:
%   dDBFH: uses the existing hard-anchor solve-for-E2 operator
%          diff_dbfh_prepare_system(_gpu), Hutchinson right preconditioning,
%          real-stacked LSQR, and diff_dbfh_unpack_solution(_gpu).
%
%   dOTF : uses two PSF images: reference pupil E1 and modified/full pupil
%          E1+E2. It computes dOTF = IFFT(I_full - I_ref), extracts one
%          non-overlapped Codona pupil copy, and compares only on the valid
%          dOTF support. This is deliberate: single-probe dOTF does not
%          recover the obscured overlap/mini-OTF region.
%
% Run from the project root containing:
%   ImZ_from_ReZ/    diff_DBFH/    fftc/
%
% 20 May 2026

clear; close all; clc;

addpath('ImZ_from_ReZ');
addpath('diff_DBFH');
addpath('fftc');

rng(1);

%% ========================================================================
%  USER PARAMETERS
%  ========================================================================

% HWO-like segmented aperture, copied from demo_differential_DBFH_gpu.m.
N                 = 2048/2;
seg_radius_pix    = 40/2;
num_rings         = 3;
string_depth_pix  = 10/2;
gap_pix           = 1;

% Noise model: independent additive Gaussian noise on each PSF image,
% scaled by max(I_full(:)). For debug, use N=512 and noise_level=0 first.
noise_level       = 1e-4;
use_noisy_mag1_for_dDBFH = true;     % true = fairer image-based comparison

% Cross-term recovery controls for dDBFH.
d_theta           = 0.25;
theta_list        = 0:d_theta:180-d_theta;
use_exact_ReZ     = false;           % false: ReZ = DeltaI/2 approximation
use_true_Z_solver = false;           % true: bypass KK and feed Z_true

% Solver controls, copied from the working GPU demo.
useGPU            = false;
nsamp             = 30;
blockSize         = 1;
precond_cap       = 1e2;
lsqr_tol          = 1e-8;
lsqr_maxit        = 1e3;
K_threshold       = 1e-3;
supportScale      = 1;
crossWeight       = 100;

% dOTF extraction. With a finite cap there is no exact single delta location;
% the centroid is used. Set dotf_anchor_search_px>0 only for debugging the
% discrete shift convention, because it uses the simulated truth to choose
% the best integer probe location.
dotf_anchor_search_px = 0;

show_plots        = true;
save_results      = true;
out_dir           = fullfile(pwd, 'results_dDBFH_vs_dOTF');

if save_results && ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% ========================================================================
%  BUILD SEGMENTED APERTURE AND DIFFERENTIAL SUPPORTS
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

####   OREN: Stopped here - experiment data came in. was fixing variable names


%% ========================================================================
%  FORWARD IMAGES AND CROSS TERM
%  ========================================================================

A1   = fft2c(E1);
A2   = fft2c(E2);
Atot = A1 + A2;

I_1_clean  = abs(A1).^2;        % reference PSF: cap blocked / absent
I_12_clean = abs(Atot).^2;      % modified/full PSF: cap present

noise_scale = max(I_12_clean(:));
I_1_noisy   = I_1_clean  + noise_level * noise_scale * randn(size(I_1_clean));
I_12_noisy  = I_12_clean + noise_level * noise_scale * randn(size(I_12_clean));

I_1_for_mag = I_1_clean;
if use_noisy_mag1_for_dDBFH
    I_1_for_mag = I_1_noisy;
end
mag1 = sqrt(max(I_1_for_mag, 0));

Z_true    = conj(A1) .* A2;
ReZ_exact = real(Z_true);
ImZ_true  = imag(Z_true);

DeltaI_noisy = I_12_noisy - I_1_noisy;

if use_exact_ReZ
    ReZ_meas = ReZ_exact;
else
    % First-order differential approximation. The neglected |A2|^2 term is
    % part of what the comparison should expose.
    ReZ_meas = 0.5 * DeltaI_noisy;
end

CSg = correlation_support(supp1, supp2);

%% ========================================================================
%  dDBFH: RECOVER Im(Z), THEN SOLVE HARD-ANCHOR SYSTEM
%  ========================================================================

if use_true_Z_solver
    Z_solver = Z_true;
    kk = struct();
    solver_label = 'dDBFH using true Z';
else
    if ~exist('recover_ImZ_by_cartesian_KK', 'file')
        error(['recover_ImZ_by_cartesian_KK not found. Add ImZ_from_ReZ/ ', ...
               'to the path or set use_true_Z_solver=true for a solver-only check.']);
    end

    [ImZ_rec, kk] = recover_ImZ_by_cartesian_KK( ...
        ReZ_meas, CSg, theta_list, ...
        'InterpMethod', 'cubic', ...
        'trueImZ', ImZ_true);

    Z_solver = ReZ_meas + 1i*ImZ_rec;
    solver_label = 'dDBFH using recovered Z';
end

rel_ReZ_err = relnorm(ReZ_meas - ReZ_exact, ReZ_exact, true(size(ReZ_exact)));
rel_ImZ_err = relnorm(imag(Z_solver) - ImZ_true, ImZ_true, true(size(ImZ_true)));
rel_Z_err   = relnorm(Z_solver - Z_true, Z_true, true(size(Z_true)));

fprintf('\n===== Cross-term diagnostics =====\n');
fprintf('noise_level                  = %.3e\n', noise_level);
fprintf('use_noisy_mag1_for_dDBFH     = %d\n', use_noisy_mag1_for_dDBFH);
fprintf('use_exact_ReZ / true_Z        = %d / %d\n', use_exact_ReZ, use_true_Z_solver);
fprintf('relative ReZ error           = %.3e\n', rel_ReZ_err);
fprintf('relative ImZ error           = %.3e\n', rel_ImZ_err);
fprintf('relative complex Z error     = %.3e\n', rel_Z_err);

fprintf('\n===== Running dDBFH =====\n');
ddb_opts = struct();
ddb_opts.useGPU       = useGPU;
ddb_opts.nsamp        = nsamp;
ddb_opts.blockSize    = blockSize;
ddb_opts.precond_cap  = precond_cap;
ddb_opts.lsqr_tol     = lsqr_tol;
ddb_opts.lsqr_maxit   = lsqr_maxit;
ddb_opts.K_threshold  = K_threshold;
ddb_opts.supportScale = supportScale;
ddb_opts.crossWeight  = crossWeight;

ddb = run_dDBFH_solver(mag1, Z_solver, supp1, supp2, E, E1, E2, supp12, ddb_opts);

%% ========================================================================
%  dOTF: TWO PSFs -> OTF DIFFERENCE -> EXTRACT NON-OVERLAPPED COPY
%  ========================================================================

fprintf('\n===== Running dOTF =====\n');
dotf_opts = struct();
dotf_opts.anchor_search_px = dotf_anchor_search_px;
dotf = run_dOTF_single_probe(I_1_noisy, I_12_noisy, E, supp1, supp2, X, Y, dotf_opts);

%% ========================================================================
%  SIDE-BY-SIDE METRICS
%  ========================================================================

valid_dotf = dotf.valid;

% Evaluate dDBFH on full pupil and also on exactly the dOTF-valid support.
ddb_full = metric_pack(E, ddb.E_rec_scaled, supp12, ddb.solve_time, ddb.iter, ddb.relres);
ddb_on_dotf = metric_pack(E, ddb.E_rec_scaled, valid_dotf, ddb.solve_time, ddb.iter, ddb.relres);
dotf_valid = metric_pack(E, dotf.E_rec_scaled, valid_dotf, dotf.solve_time, NaN, NaN);

Method = {'dDBFH full pupil'; 'dDBFH on dOTF-valid pixels'; 'dOTF single-probe valid pixels'};
CoveragePct = 100 * [nnz(supp12)/nnz(supp12); nnz(valid_dotf)/nnz(supp12); nnz(valid_dotf)/nnz(supp12)];
PhaseRMSE_rad = [ddb_full.phase_rmse; ddb_on_dotf.phase_rmse; dotf_valid.phase_rmse];
FieldRelErr_scalar = [ddb_full.field_relerr; ddb_on_dotf.field_relerr; dotf_valid.field_relerr];
AmpRelErr_scalar = [ddb_full.amp_relerr; ddb_on_dotf.amp_relerr; dotf_valid.amp_relerr];
Time_s = [ddb_full.time_s; ddb_on_dotf.time_s; dotf_valid.time_s];
LSQRIter = [ddb_full.iter; ddb_on_dotf.iter; dotf_valid.iter];
LSQRRelres = [ddb_full.relres; ddb_on_dotf.relres; dotf_valid.relres];

metrics = table(Method, CoveragePct, PhaseRMSE_rad, FieldRelErr_scalar, ...
    AmpRelErr_scalar, Time_s, LSQRIter, LSQRRelres);

disp(metrics);

fprintf('\nInterpretation check:\n');
fprintf('  dOTF valid coverage = %.2f %% of supp12. Do not compare its valid-mask RMSE\n', ...
    100*nnz(valid_dotf)/nnz(supp12));
fprintf('  against dDBFH full-pupil RMSE without also looking at dDBFH-on-dOTF-valid.\n');
fprintf('  A single dOTF perturbation has a real blind/overlap region; dDBFH is being\n');
fprintf('  asked to reconstruct the full E1+E2 pupil.\n');

results = struct();
results.metrics = metrics;
results.ddb = ddb;
results.dotf = dotf;
results.params = struct('N',N,'seg_radius_pix',seg_radius_pix,'num_rings',num_rings, ...
    'string_depth_pix',string_depth_pix,'gap_pix',gap_pix,'noise_level',noise_level, ...
    'use_noisy_mag1_for_dDBFH',use_noisy_mag1_for_dDBFH,'use_exact_ReZ',use_exact_ReZ, ...
    'use_true_Z_solver',use_true_Z_solver,'K_threshold',K_threshold, ...
    'crossWeight',crossWeight,'dotf_anchor_search_px',dotf_anchor_search_px);
results.truth = struct('E',E,'E1',E1,'E2',E2,'supp1',supp1,'supp2',supp2, ...
    'supp12',supp12,'gap_mask',gap_mask,'seg_id',seg_id,'seg_centers',seg_centers);
results.cross_term = struct('Z_true',Z_true,'Z_solver',Z_solver,'ReZ_meas',ReZ_meas, ...
    'rel_ReZ_err',rel_ReZ_err,'rel_ImZ_err',rel_ImZ_err,'rel_Z_err',rel_Z_err,'kk',kk);

if save_results
    save(fullfile(out_dir, 'dDBFH_vs_dOTF_HWO_results.mat'), 'results', '-v7.3');
    writetable(metrics, fullfile(out_dir, 'dDBFH_vs_dOTF_HWO_metrics.csv'));
end

%% ========================================================================
%  PLOTS
%  ========================================================================

if show_plots
    plot_supports(supp1, supp2, gap_mask, valid_dotf, CSg);
    plot_field_comparison(E, ddb.E_rec_scaled, supp12, [solver_label ' full pupil']);
    plot_field_comparison(E, ddb.E_rec_scaled, valid_dotf, 'dDBFH evaluated only on dOTF-valid mask');
    plot_field_comparison(E, dotf.E_rec_scaled, valid_dotf, 'dOTF single-probe reconstruction, valid mask only');
    plot_dotf_products(dotf);
end

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function out = run_dDBFH_solver(mag1, Z_solver, supp1, supp2, E, E1, E2, supp12, opts)
%RUN_DDBFH_SOLVER Hard-anchor solve-for-E2 dDBFH pipeline.
addpath('ImZ_from_ReZ');
addpath('diff_DBFH');
addpath('fftc');

if opts.useGPU
    assert(gpuDeviceCount > 0, 'useGPU=true but no CUDA GPU is available.');
    gd = gpuDevice;
    fprintf('Using GPU: %s, available memory %.2f GB\n', gd.Name, gd.AvailableMemory/1024^3);

    mag1_sys = gpuArray(mag1);
    Z_sys    = gpuArray(Z_solver);
    supp1_sys = gpuArray(supp1);
    supp2_sys = gpuArray(supp2);

    sys = diff_dbfh_prepare_system_gpu( ...
        mag1_sys, Z_sys, supp1_sys, supp2_sys, ...
        'lambda', 0, ...
        'lambdaX1', 0, ...
        'lambdaE2', 0, ...
        'mag1RelThresh', opts.K_threshold, ...
        'cRelThresh', 0, ...
        'supportScale', opts.supportScale, ...
        'crossWeight', opts.crossWeight, ...
        'anchorVal', 1, ...
        'anchorStrategy', 'maxMag1');
else
    sys = diff_dbfh_prepare_system( ...
        mag1, Z_solver, supp1, supp2, ...
        'lambda', 0, ...
        'lambdaX1', 0, ...
        'lambdaE2', 0, ...
        'mag1RelThresh', opts.K_threshold, ...
        'cRelThresh', 0, ...
        'supportScale', opts.supportScale, ...
        'crossWeight', opts.crossWeight, ...
        'anchorVal', 1, ...
        'anchorStrategy', 'maxMag1');
end

fprintf('sys.n = %d = %d X1-free + %d E2 pixels\n', sys.n, sys.nX1, sys.nE2);
fprintf('valid focal K = %d pixels; anchor = %d\n', sys.nK, sys.anchor_idx);

fprintf('Building Hutchinson right preconditioner...\n');
d = hutchinson_diag_AtA(sys, opts.nsamp, opts.blockSize, opts.useGPU);
d = d(:);
d_cpu = to_cpu(d);
dpos = d_cpu(d_cpu > 0 & isfinite(d_cpu));
assert(~isempty(dpos), 'Hutchinson diagonal has no positive finite entries.');

dfloor = 1e-8 * median(dpos);
p0 = 1 ./ sqrt(max(d, dfloor));
p0_cpu = to_cpu(p0);
p0pos = p0_cpu(isfinite(p0_cpu) & p0_cpu > 0);
p_med = median(p0pos);
p_cap = opts.precond_cap * p_med;
p = min(p0, p_cap);

Afun = make_Afun_lsqr_stacked_diagprecond(sys, p);
bstack = [real(sys.b); imag(sys.b)];

fprintf('Solving real-stacked LSQR...\n');
tic;
[ustack, flag, relres, iter, resvec] = lsqr(Afun, bstack, opts.lsqr_tol, opts.lsqr_maxit);
solve_time = toc;

ur = ustack(1:sys.n);
ui = ustack(sys.n+1:end);
xvec = p .* (ur + 1i*ui);

if opts.useGPU
    unpack = diff_dbfh_unpack_solution_gpu(sys, xvec, 'normalizeX1', false);
    unpack = gather_struct(unpack);
    sys_small = gather_sys_nohandles(sys);
else
    unpack = diff_dbfh_unpack_solution(sys, xvec, 'normalizeX1', false);
    sys_small = gather_sys_nohandles(sys);
end

E_rec = unpack.E;
[~, E_rec_scaled, alpha] = align_complex_scalar(E, E_rec, supp12);
E_rec_phase = align_global_phase(E, E_rec, supp12);

out = struct();
out.method = 'dDBFH';
out.sys = sys_small;
out.unpack = unpack;
out.E_rec = E_rec;
out.E_rec_scaled = E_rec_scaled;
out.E_rec_phase = E_rec_phase;
out.alpha = alpha;
out.flag = flag;
out.relres = double(to_cpu(relres));
out.iter = iter;
out.resvec = to_cpu(resvec);
out.solve_time = solve_time;
out.leak1 = unpack.leak1;
out.cross_relres = unpack.cross_relres;
out.stats_full = metric_pack(E, E_rec_scaled, supp12, solve_time, iter, out.relres);
out.stats_E1 = metric_pack(E1, E_rec_scaled, supp1, solve_time, iter, out.relres);
out.stats_E2 = metric_pack(E2, E_rec_scaled, supp2, solve_time, iter, out.relres);

fprintf('dDBFH flag/relres/iter/time = %d / %.3e / %d / %.2f s\n', ...
    flag, out.relres, iter, solve_time);
fprintf('dDBFH full phase RMSE / scalar relerr = %.3e rad / %.3e\n', ...
    out.stats_full.phase_rmse, out.stats_full.field_relerr);
end

function dotf = run_dOTF_single_probe(I_ref, I_mod, Etrue, supp1, supp2, X, Y, opts)
%RUN_DOTF_SINGLE_PROBE Codona-style two-image dOTF extraction.
% I_ref: PSF for E1. I_mod: PSF for E1+E2.

search_px = 0;
if isfield(opts, 'anchor_search_px')
    search_px = opts.anchor_search_px;
end

fprintf('Computing dOTF = IFFT(I_mod - I_ref)...\n');
tic;
dO = ifft2c(I_mod - I_ref);

x0_base = round(mean(X(supp2)));
y0_base = round(mean(Y(supp2)));

CS12 = correlation_support(supp1, supp2);  % support of IFT[A1^* A2]
CS21 = correlation_support(supp2, supp1);  % support of IFT[A2^* A1]
CS22 = correlation_support(supp2, supp2);  % mini-OTF support

best = [];
for dx0 = -search_px:search_px
    for dy0 = -search_px:search_px
        x0 = x0_base + dx0;
        y0 = y0_base + dy0;

        cand1 = extract_dotf_copy(dO, Etrue, supp1, X, Y, x0, y0, CS12, CS21, CS22, 'plus');
        cand2 = extract_dotf_copy(dO, Etrue, supp1, X, Y, x0, y0, CS12, CS21, CS22, 'minus');

        if isempty(best) || cand1.phase_rmse < best.phase_rmse
            best = cand1;
        end
        if cand2.phase_rmse < best.phase_rmse
            best = cand2;
        end
    end
end
solve_time = toc;

if isempty(best) || nnz(best.valid) == 0
    error('dOTF extraction produced an empty valid mask. Check support geometry/FFT convention.');
end

dotf = best;
dotf.method = 'dOTF';
dotf.dOTF = dO;
dotf.CS12 = CS12;
dotf.CS21 = CS21;
dotf.CS22 = CS22;
dotf.solve_time = solve_time;
dotf.anchor_base = [x0_base, y0_base];

fprintf('dOTF selected copy = %s, anchor [x0,y0]=[%d,%d], valid pixels=%d\n', ...
    dotf.copy_name, dotf.x0, dotf.y0, nnz(dotf.valid));
fprintf('dOTF coverage / phase RMSE / scalar relerr = %.2f %% / %.3e rad / %.3e\n', ...
    100*nnz(dotf.valid)/nnz(supp1 | supp2), dotf.phase_rmse, dotf.field_relerr);
end

function cand = extract_dotf_copy(dO, Etrue, supp1, X, Y, x0, y0, CS12, CS21, CS22, mode)
%EXTRACT_DOTF_COPY Extract one of the two conjugate/reflected dOTF copies.

[Nr,Nc] = size(dO);
cx = round((Nc+1)/2);
cy = round((Nr+1)/2);

switch lower(mode)
    case 'plus'
        % IFT[A1^* A2] lives at displacement s = x2 - x1 and is conjugated.
        sx = x0 - X;
        sy = y0 - Y;
        copy_name = 'IFT[A1^*A2] conjugated copy';
        use_conj = true;
        ownCS = CS12;
        otherCS = CS21;
    case 'minus'
        % IFT[A2^* A1] lives at displacement s = x1 - x2.
        sx = X - x0;
        sy = Y - y0;
        copy_name = 'IFT[A2^*A1] direct copy';
        use_conj = false;
        ownCS = CS21;
        otherCS = CS12;
    otherwise
        error('Unknown dOTF copy mode: %s', mode);
end

col = cx + sx;
row = cy - sy;
in = row >= 1 & row <= Nr & col >= 1 & col <= Nc;

idx = zeros(Nr,Nc);
idx(in) = sub2ind([Nr,Nc], row(in), col(in));

raw = zeros(Nr,Nc);
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
mini_at(in) = CS22(idx(in));

valid = supp1 & in & own_at & ~other_at & ~mini_at & isfinite(real(raw)) & isfinite(imag(raw));

if nnz(valid) > 0
    [~, E_rec_scaled, alpha] = align_complex_scalar(Etrue, raw, valid);
    phase_rmse_val = phase_rmse(Etrue, E_rec_scaled, valid);
    field_relerr_val = relnorm(E_rec_scaled - Etrue, Etrue, valid);
    amp_relerr_val = relnorm(abs(E_rec_scaled) - abs(Etrue), abs(Etrue), valid);
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

function Frec_phase = align_global_phase(Ftrue, Frec, mask)
mask = logical(mask) & abs(Ftrue) > 0 & abs(Frec) > 0;
if ~any(mask(:))
    Frec_phase = Frec;
    return;
end
phi0 = angle(sum(conj(Ftrue(mask)) .* Frec(mask)));
Frec_phase = Frec .* exp(-1i*phi0);
end

function CSg = correlation_support(E1_mask, E2_mask)
%CORRELATION_SUPPORT Support of IFT[A1^* A2], using the same support logic
%as demo_differential_DBFH_gpu.m.
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
mask = logical(mask) & isfinite(real(err)) & isfinite(imag(err)) & isfinite(real(ref)) & isfinite(imag(ref));
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
% Keep scalar/mask diagnostics but avoid saving function handles/operators.
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

function plot_supports(supp1, supp2, gap_mask, valid_dotf, CSg)
figure('Color','w','Name','Supports','Position',[100 100 1500 450]);
tiledlayout(1,5,'TileSpacing','compact','Padding','compact');
nexttile; imagesc(supp1); axis image off; colorbar; title('supp1 / reference pupil');
nexttile; imagesc(supp2); axis image off; colorbar; title('supp2 / perturbing cap');
nexttile; imagesc(gap_mask); axis image off; colorbar; title('intentional gap');
nexttile; imagesc(valid_dotf); axis image off; colorbar; title('dOTF valid eval mask');
nexttile; imagesc(CSg); axis image off; colorbar; title('CSg = supp IFT[A1^*A2]');
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

function plot_dotf_products(dotf)
figure('Color','w','Name','dOTF products','Position',[100 100 1650 850]);
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
