%% demo_differential_DBFH_HWO_geometry_clean.m
% Clean differential-DBFH demo for a segmented HWO-like aperture.
%
% Purpose
%   1. Build a segmented complex pupil field E = E1 + E2.
%   2. Form the differential cross term Z = A1^* A2.
%   3. Optionally recover Im(Z) from Re(Z) using the Cartesian KK routine.
%   4. Solve the differential DBFH linear system for X1.
%   5. Compare reconstructed pupil fields against truth with annotated figures.
%
% Critical convention
%   All focal-plane arrays are FFT-centered. The helper functions fft2c/ifft2c
%   below use fftshift/ifftshift and unitary normalization.
%
% Critical solver detail
%   make_Afun_lsqr_stacked_diagprecond solves for the preconditioned variable u.
%   The physical phasor vector is x = D*u, i.e. x1vec = p .* u_complex.
% 
% Status: 
%   Solver fails when applying the approximation delta I = 2 Re(conj(A1).*A2)
%   because the denominator X/|A1| amplifies biases and errors when |A1| is
%   small. For that reason, we would like to solve for the A2 field (not just
%   X2). This is done in a similar file with suffix A2.

clear; close all; clc;

addpath('diff_DBFH\');
addpath('sign_problem\');

rng(1);

%% ========================================================================
%  USER PARAMETERS
%  ========================================================================

N                 = 512;          % grid size
seg_radius_pix    = 20;            % hex circumradius in pixels
num_rings         = 3;               % segmented aperture rings
string_depth_pix  = 1;            % depth of the E2 cap inside the top segment
gap_pix           = 1;               % physical gap between E1 and E2 supports
noise_level       = 0*1e-3;              % gaussian noise

d_theta           = 0.1;
theta_list        = 0:d_theta:180-d_theta;

use_exact_ReZ     = false;           % false: ReZ = DeltaI/2, true: ReZ = real(Z_true)
use_true_Z_solver = false;            % true: validate DBFH using exact Z_true
                                      % false: solve using Z_rec = ReZ_meas + 1i*ImZ_rec
show_plots        = true;
run_adjoint_tests = true;

% LSQR / preconditioner controls
nsamp             = 30;              % Hutchinson samples
blockSize         = 1;
useGPU            = false;
precond_cap       = 1e2;             % cap p <= precond_cap * median(p)
lsqr_tol          = 1e-12;
lsqr_maxit        = 5000;
K_threshold       = 1e-3;

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

% E2 is a thin outer cap. E1 is the rest of the aperture excluding a gap.
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

% DeltaI = I_12 - I_1;
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
%  BUILD DIFFERENTIAL DBFH SYSTEM
%  ========================================================================

if use_true_Z_solver
    Z_solver = Z_true;
    solver_label = 'true Z = A1^*A2';
else
    Z_solver = Z_rec;
    solver_label = 'recovered Z = ReZ_meas + i ImZ_rec';
end

% Use the scaled builder if it exists. Fall back to the base builder.
if exist('diff_dbfh_prepare_system_scaled', 'file') == 2
    sys = diff_dbfh_prepare_system_scaled( ...
        mag1, Z_solver, supp1, supp2, ...
        'lambda', 0, ...
        'mag1RelThresh', K_threshold, ...
        'cRelThresh', 0, ...
        'supportScale', sqrt(numel(mag1)), ...
        'anchorWeight', 1);
else
    sys = diff_dbfh_prepare_system( ...
        mag1, Z_solver, supp1, supp2, ...
        'lambda', 0, ...
        'mag1RelThresh', K_threshold, ...
        'cRelThresh', 0);
end

% Align true X1 to the solver anchor, then test the true vector through sys.
X1_true_aligned = X1_true;
if isfield(sys, 'anchor_idx') && sys.anchor_idx > 0
    anchor_phase = X1_true(sys.anchor_idx);
    X1_true_aligned = X1_true / anchor_phase;
end

xtrueK = X1_true_aligned(sys.K);
rtrue = sys.Amul(xtrueK) - sys.b;
true_sys_relres = norm(rtrue) / max(norm(sys.b), eps);
out_trueK = diff_dbfh_unpack_solution(sys, xtrueK, 'normalizePhasor', false);

fprintf('\n===== System feasibility check =====\n');
fprintf('solver cross term             = %s\n', solver_label);
fprintf('TRUE X1 through sys relres    = %.3e\n', true_sys_relres);
fprintf('TRUE X1 through sys leaks     = %.3e / %.3e\n', out_trueK.leak1, out_trueK.leak2);

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

p_med = median(p0(isfinite(p0) & p0 > 0));
p_cap = precond_cap * p_med;
p = min(p0, p_cap);

fprintf('p length/sys.n = %d / %d\n', numel(p), sys.n);
fprintf('p uncapped range = %.3e / %.3e\n', min(p0), max(p0));
fprintf('p capped range   = %.3e / %.3e\n', min(p), max(p));
fprintf('cap factor       = %.3e\n', precond_cap);

Afun = make_Afun_lsqr_stacked_diagprecond(sys, p);
bstack = [real(sys.b); imag(sys.b)];

if run_adjoint_tests
    [stack_adj_err, true_stack_relres] = test_stacked_operator(Afun, sys, xtrueK, p, bstack);
    fprintf('\n===== Real-stacked operator checks =====\n');
    fprintf('TRUE through STACKED Afun relres = %.3e\n', true_stack_relres);
    fprintf('STACKED adjoint rel err          = %.3e\n', stack_adj_err);
end

fprintf('\n===== LSQR solve =====\n');
tic;
[ustack, flag, relres, iter, resvec] = lsqr(Afun, bstack, lsqr_tol, lsqr_maxit);
solve_time = toc;

% IMPORTANT: LSQR solved for u in A*D*u = b. Convert back to physical x.
ur = ustack(1:sys.n);
ui = ustack(sys.n+1:end);
x1vec = p .* (ur + 1i*ui);

fprintf('flag/relres/iter             = %d / %.3e / %d\n', flag, relres, iter);
fprintf('solve time                   = %.3f s\n', solve_time);
fprintf('|x| median/min/max           = %.3e / %.3e / %.3e\n', ...
    median(abs(x1vec)), min(abs(x1vec)), max(abs(x1vec)));
fprintf('|x| p01/p99                  = %.3e / %.3e\n', ...
    percentile(abs(x1vec), 1), percentile(abs(x1vec), 99));

%% ========================================================================
%  UNPACK AND ALIGN RECONSTRUCTION
%  ========================================================================

out_raw = diff_dbfh_unpack_solution(sys, x1vec, 'normalizePhasor', false);
out_norm = diff_dbfh_unpack_solution(sys, x1vec, 'normalizePhasor', true);

% Use raw unless you know the linear solution already has |X1| very close to 1.
out = out_raw;

X1_rec = out.X1;
X2_rec = out.X2;
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
stats.raw_leak2        = out_raw.leak2;
stats.norm_leak1       = out_norm.leak1;
stats.norm_leak2       = out_norm.leak2;
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

fprintf('\n===== Reconstruction metrics =====\n');
fprintf('raw leaks leak1/leak2         = %.3e / %.3e\n', stats.raw_leak1, stats.raw_leak2);
fprintf('norm leaks leak1/leak2        = %.3e / %.3e\n', stats.norm_leak1, stats.norm_leak2);
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
    plot_solver_diagnostics(sys, p, d, resvec, x1vec, X1_true_aligned);

    plot_complex_field_comparison(E, E_rec_aligned, supp12, ...
        sprintf('Full pupil reconstruction: %s', solver_label));

    plot_phase_error_only(E, E_rec_aligned, supp12, ...
        sprintf('Final phase error, RMSE = %.3e rad', stats.phase_rmse_rad));
end

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

function F = fft2c(x)
%FFT2C Centered, unitary 2D FFT.
F = fftshift(fft2(ifftshift(x))) / sqrt(numel(x));
end

function x = ifft2c(F)
%IFFT2C Centered, unitary inverse 2D FFT.
x = fftshift(ifft2(ifftshift(F))) * sqrt(numel(F));
end

function F = fft1c(x)
%FFT1C Centered 1D FFT. Non-unitary to match the KK prototype.
F = fftshift(fft(ifftshift(x)));
end

function x = ifft1c(F)
%IFFT1C Centered 1D inverse FFT. Non-unitary to match the KK prototype.
x = fftshift(ifft(ifftshift(F)));
end

function Afun = make_Afun_lsqr_stacked_diagprecond(sys, p)
%MAKE_AFUN_LSQR_STACKED_DIAGPRECOND Real-stacked LSQR wrapper.
%
% Complex system:
%   A z = b
%
% Right-preconditioned real system:
%   A D u = b,    z = D u
%
% The vector p stores diag(D). LSQR returns u, not z.

n = sys.n;
m = sys.m;

p = double(gather(p(:)));
assert(numel(p) == n, 'p must have length sys.n.');
assert(all(isfinite(p)), 'p contains NaN or Inf.');
assert(all(p > 0), 'p must be strictly positive.');

Afun = @afun_local;

    function y = afun_local(x, transp_flag)
        switch transp_flag
            case 'notransp'
                ur = x(1:n);
                ui = x(n+1:end);
                z = p .* (ur + 1i*ui);

                Az = sys.Amul(z);
                y = [real(Az); imag(Az)];

            case 'transp'
                yr = x(1:m);
                yi = x(m+1:end);
                yc = yr + 1i*yi;

                g = sys.ATmul(yc);
                g = p .* g;   % adjoint of A*D is D*A'

                y = [real(g); imag(g)];

            otherwise
                error('Unknown transpose flag: %s', transp_flag);
        end
    end
end

function [adj_err, true_relres] = test_stacked_operator(Afun, sys, xtrueK, p, bstack)
%TEST_STACKED_OPERATOR Check preconditioned real-stacked feasibility and adjointness.
% Afun represents A*D, where D = diag(p). Therefore the true
% preconditioned variable is u_true = D^{-1} x_true.

utrueK = xtrueK ./ p(:);
utrue_stack = [real(utrueK); imag(utrueK)];
rstack = Afun(utrue_stack, 'notransp') - bstack;
true_relres = norm(rstack) / max(norm(bstack), eps);

x = randn(2*sys.n,1);
y = randn(2*sys.m,1);

Ax  = Afun(x, 'notransp');
ATy = Afun(y, 'transp');

lhs = y(:)' * Ax(:);
rhs = ATy(:)' * x(:);
adj_err = abs(lhs-rhs) / max(abs(lhs), abs(rhs));

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

function q = percentile(x, pct)
%PERCENTILE Lightweight percentile helper without toolbox dependency.
x = sort(x(:));
x = x(isfinite(x));
if isempty(x)
    q = NaN;
    return;
end
idx = 1 + (numel(x)-1) * pct/100;
lo = floor(idx);
hi = ceil(idx);
if lo == hi
    q = x(lo);
else
    q = x(lo) + (idx-lo) * (x(hi)-x(lo));
end
end

function plot_support_diagnostics(supp1, supp2, gap_mask, CSg)
%PLOT_SUPPORT_DIAGNOSTICS Show pupil supports and correlation support.

figure('Color','w','Name','Support diagnostics','Position',[100 100 1500 450]);
tiledlayout(1,4,'TileSpacing','compact','Padding','compact');

nexttile; imagesc(supp1); axis image off; colorbar; title('supp1: exposed pupil');
nexttile; imagesc(supp2); axis image off; colorbar; title('supp2: blocked cap');
nexttile; imagesc(gap_mask); axis image off; colorbar; title('intentional gap');
nexttile; imagesc(CSg); axis image off; colorbar; title('CSg = supp IFT[A1^*A2]');
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
    nexttile; plot(real(kk.gI_true_all(:,ia)),'LineWidth',1.2); hold on; plot(real(kk.gI_all(:,ia)),'--','LineWidth',1.2); grid on; title('Re line: truth vs rec'); legend('truth','rec');
    nexttile; plot(imag(kk.gI_true_all(:,ia)),'LineWidth',1.2); hold on; plot(imag(kk.gI_all(:,ia)),'--','LineWidth',1.2); grid on; title('Im line: truth vs rec'); legend('truth','rec');
    nexttile; imagesc(real(kk.gI_all - kk.gI_true_all)); axis image; colorbar; title('Re line-domain error');
    nexttile; imagesc(imag(kk.gI_all - kk.gI_true_all)); axis image; colorbar; title('Im line-domain error');
    colormap parula;
end

end

function plot_solver_diagnostics(sys, p, d, resvec, x1vec, X1_true_aligned)
%PLOT_SOLVER_DIAGNOSTICS Inspect preconditioner, residual convergence, and X1.

P = nan(size(sys.K));
P(sys.K) = p;
D = nan(size(sys.K));
D(sys.K) = d;
X = nan(size(sys.K));
X(sys.K) = abs(x1vec);

phase_err = nan(size(sys.K));
if ~isempty(X1_true_aligned)
    Xrec_full = nan(size(sys.K));
    Xrec_full(sys.K) = x1vec;
    phase_err(sys.K) = angle(conj(X1_true_aligned(sys.K)) .* Xrec_full(sys.K));
end

figure('Color','w','Name','Solver diagnostics','Position',[100 100 1600 850]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile; semilogy(resvec,'LineWidth',1.2); grid on; xlabel('iteration'); ylabel('residual'); title('LSQR residual history');
nexttile; imagesc(log10(D)); axis image off; colorbar; title('log10 Hutchinson diag(A^*A)');
nexttile; imagesc(log10(P)); axis image off; colorbar; title('log10 right preconditioner p');
nexttile; imagesc(X); axis image off; colorbar; title('|X1 recovered| on K');
nexttile; imagesc(phase_err); axis image off; colorbar; clim([-pi pi]); title('phase error of X1 on K');
nexttile; histogram(abs(x1vec),100); grid on; xlabel('|x1vec|'); ylabel('count'); title('|X1| histogram');
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

function plot_phase_error_only(Ftrue, Frec, mask, figTitle)
%PLOT_PHASE_ERROR_ONLY Final compact phase-error plot.

phase_err = angle(conj(Ftrue).*Frec);

figure('Color','w','Name','Final phase error','Position',[200 200 700 600]);
imagesc_masked(phase_err, mask);
axis image off; colorbar; clim([-pi pi]);
title(figTitle, 'Interpreter','none');
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

function [ImZ_rec, info] = recover_ImZ_by_cartesian_KK(ReZ, CSg, theta_list, varargin)
%RECOVER_IMZ_BY_CARTESIAN_KK Recover Im(Z) from Re(Z) using a Cartesian
%line-domain one-sided support construction.
%
% This avoids MATLAB radon bin-center conventions. It computes
%   gR_cart = ifft2c(ReZ)
% directly, samples Cartesian radial lines, imposes one-sided support, and
% grids the recovered line data back to a Cartesian correlation plane.

p = inputParser;
p.addParameter('KKSign', +1, @(x) isnumeric(x) && isscalar(x) && any(x == [-1,+1]));
p.addParameter('SideFlip', +1, @(x) isnumeric(x) && isscalar(x) && any(x == [-1,+1]));
p.addParameter('YSign', +1, @(x) isnumeric(x) && isscalar(x) && any(x == [-1,+1]));
p.addParameter('InterpMethod', 'linear', @(x) ischar(x) || isstring(x));
p.addParameter('trueImZ', [], @(x) isempty(x) || isequal(size(x), size(ReZ)));
p.parse(varargin{:});

KKSign      = p.Results.KKSign;
SideFlip    = p.Results.SideFlip;
YSign       = p.Results.YSign;
interpMethod = char(p.Results.InterpMethod);
trueImZ     = p.Results.trueImZ;

N = size(ReZ,1);
assert(size(ReZ,1) == size(ReZ,2), 'Use square grids for this prototype.');

% FFT-centered Cartesian coordinates: x rightward, y upward.
c = floor(N/2) + 1;
x = (1:N) - c;
y = c - (1:N);
[Xcart, Ycart] = meshgrid(x, y);
s = x(:);

num_rho    = numel(s);
num_angles = numel(theta_list);

gR_cart = ifft2c(ReZ);
[FR, FI] = make_complex_interpolants(gR_cart, x, y, interpMethod);

have_true = ~isempty(trueImZ);
if have_true
    gI_true_cart = ifft2c(trueImZ);
    [FTR, FTI] = make_complex_interpolants(gI_true_cart, x, y, interpMethod);
    gI_true_all = zeros(num_rho, num_angles);
else
    gI_true_all = [];
end

% Estimate the one-sided support direction from the centroid of CSg.
[rr, cc] = find(CSg);
gx = cc - c;
gy = c  - rr;
cent = [mean(gx), mean(gy)];
n_hat = cent / max(norm(cent), eps);

gR_all = zeros(num_rho, num_angles);
gI_all = zeros(num_rho, num_angles);
R_Re   = zeros(num_rho, num_angles);
R_Im   = zeros(num_rho, num_angles);
valid_frac = zeros(1, num_angles);

for ia = 1:num_angles
    theta = theta_list(ia);

    xline = s .* cosd(theta);
    yline = YSign .* s .* sind(theta);

    gR = FR(yline, xline) + 1i * FI(yline, xline);
    gR(isnan(gR)) = 0;

    th = [cosd(theta), YSign * sind(theta)];
    side_sign = sign(dot(n_hat, th));
    if side_sign == 0
        continue;
    end

    side_sign = SideFlip * side_sign;

    zero_side    = side_sign * s < 0;
    support_side = side_sign * s > 0;

    gI = zeros(num_rho, 1);

    % On the zero side, g = gR + i*gI = 0, so gI = i*gR up to sign.
    gI(zero_side) = KKSign * 1i * gR(zero_side);

    % Hermitian fill: if ImZ is real, ifft2c(ImZ) is Hermitian.
    for j = 1:num_rho
        if support_side(j)
            sm = -s(j);
            [~, jm] = min(abs(s - sm));
            gI(j) = conj(gI(jm));
        end
    end

    [~, j0] = min(abs(s));
    gI(j0) = real(gI(j0));

    gR_all(:,ia) = gR;
    gI_all(:,ia) = gI;

    R_Re(:,ia) = real(fft1c(gR));
    R_Im(:,ia) = real(fft1c(gI));

    valid_frac(ia) = nnz(zero_side) / numel(zero_side);

    if have_true
        gIt = FTR(yline, xline) + 1i * FTI(yline, xline);
        gIt(isnan(gIt)) = 0;
        gI_true_all(:,ia) = gIt;
    end
end

% Grid recovered radial slices back to Cartesian correlation plane.
[ThetaGrid, SGrid] = meshgrid(deg2rad(theta_list), s);
Xp = SGrid .* cos(ThetaGrid);
Yp = YSign .* SGrid .* sin(ThetaGrid);

gI_cart_real = griddata(Xp(:), Yp(:), real(gI_all(:)), Xcart, Ycart, interpMethod);
gI_cart_imag = griddata(Xp(:), Yp(:), imag(gI_all(:)), Xcart, Ycart, interpMethod);

gI_cart = gI_cart_real + 1i * gI_cart_imag;
gI_cart(isnan(gI_cart)) = 0;

Rmax = max(abs(s));
valid_disk = sqrt(Xcart.^2 + Ycart.^2) <= Rmax;
gI_cart(~valid_disk) = 0;

ImZ_complex = fft2c(gI_cart);
ImZ_rec = real(ImZ_complex);

info = struct();
info.R_Re = R_Re;
info.R_Im = R_Im;
info.gR_all = gR_all;
info.gI_all = gI_all;
info.gI_cart = gI_cart;
info.gR_cart = gR_cart;
info.ImZ_complex = ImZ_complex;
info.s = s;
info.x = x;
info.y = y;
info.Xcart = Xcart;
info.Ycart = Ycart;
info.valid_disk = valid_disk;
info.valid_angle_fraction = valid_frac;
info.KKSign = KKSign;
info.SideFlip = SideFlip;
info.YSign = YSign;
info.n_hat = n_hat;
info.interpMethod = interpMethod;

if have_true
    info.gI_true_all = gI_true_all;
    info.line_rel_err = norm(gI_all(:) - gI_true_all(:)) / max(norm(gI_true_all(:)), eps);
else
    info.gI_true_all = [];
    info.line_rel_err = NaN;
end

end

function [FR, FI] = make_complex_interpolants(V, x, y, interpMethod)
%MAKE_COMPLEX_INTERPOLANTS Build real/imag griddedInterpolants.
% griddedInterpolant coordinates are ordered as {row-coordinate, col-coordinate}.
% y is descending in matrix row order, so flip vertically to make y ascending.

x = x(:).';
y = y(:).';

y_asc = fliplr(y);
V_asc = flipud(V);

FR = griddedInterpolant({y_asc, x}, real(V_asc), interpMethod, 'none');
FI = griddedInterpolant({y_asc, x}, imag(V_asc), interpMethod, 'none');

end
