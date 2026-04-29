
clear; close all; clc;

addpath('diff_DBFH\');
addpath('sign_problem\');

parula_with_nan_white = [1 1 1; parula(256)];

rng(1);

% ---------------- USER PARAMETERS ----------------
N = 1024/2;
seg_radius_pix = 34/2;
num_rings = 3;                 
string_depth_pix = 12/2;        
gap_pix = 1;
d_theta = 0.25;
theta_list = 0:d_theta:180-d_theta;         
use_exact_ReZ = false;         % use true ReZ first for debugging
show_plots = true;

% ---------------- BUILD SEGMENTED APERTURE ----------------
[seg_id, seg_centers, X, Y] = make_hex_segment_aperture(N, seg_radius_pix, num_rings);
aperture = seg_id > 0;
nSeg = max(seg_id(:));

% choose upper-most peripheral segment
[~, top_seg] = max(seg_centers(:,2));
top_mask = seg_id == top_seg;

% outward normal
c_top = seg_centers(top_seg,:);
n_out = c_top / norm(c_top);

S = X*n_out(1) + Y*n_out(2);
S_top = S(top_mask);
s_edge = max(S_top);

% string-cut cap
cut_E2 = s_edge - string_depth_pix;
cut_gap = cut_E2 - gap_pix;

supp2 = top_mask & (S >= cut_E2);
gap_mask = top_mask & (S >= cut_gap) & (S < cut_E2);
supp1 = aperture & ~supp2 & ~gap_mask;
supp12 = supp2+supp1;

% ---------------- RANDOM SEGMENTED PHASE FRONT ----------------
E = zeros(N,N);

for k = 1:nSeg
    mk = seg_id == k;
    if ~any(mk(:)), continue; end

    cx = seg_centers(k,1);
    cy = seg_centers(k,2);

    dx = (X - cx) / seg_radius_pix;
    dy = (Y - cy) / seg_radius_pix;

    piston = 1 * randn();
    tip    = 0.5 * randn();
    tilt   = 0.5 * randn();
    defoc  = 0.5 * randn();

    rho2 = dx.^2 + dy.^2;
    rho2_mean = mean(rho2(mk));

    phi = piston + tip*dx + tilt*dy + defoc*(rho2 - rho2_mean);

    E(mk) = exp(1i * phi(mk));
end

E1 = E .* supp1;
E2 = E .* supp2;

% ---------------- FOCAL PLANE FIELDS ----------------
A1 = fft2c(E1);
A2 = fft2c(E2);
Atot = A1 + A2;
% focal-plane phasor truth - for comparison with estimate only
X_true = Atot ./ max(abs(Atot), eps);

I_12 = abs(Atot).^2;
I_1  = abs(A1).^2;

Z_true = conj(A1) .* A2;
ReZ_exact = real(Z_true);
ImZ_true  = imag(Z_true);

DeltaI = I_12 - I_1;

if use_exact_ReZ
    ReZ_meas = ReZ_exact;
else
    % first-order approximation: neglect |A2|^2
    ReZ_meas = 0.5 * DeltaI;
end

% ---------------- SUPPORT OF g = IFT2[A1^* A2] ----------------
CSg = correlation_support(supp1, supp2);

% ---------------- KK RECOVERY OF ImZ ----------------
[ImZ_rec, kk] = recover_ImZ_by_cartesian_KK(ReZ_meas, CSg, theta_list,'InterpMethod', 'cubic');


% --- Cartesian line-domain truth check ---
if isfield(kk, 'gI_true_all') && ~isempty(kk.gI_true_all)

    gI_true = kk.gI_true_all;
    gI_rec  = kk.gI_all;

    line_err = norm(gI_rec(:) - gI_true(:)) / max(norm(gI_true(:)), eps);
    fprintf('Line-domain gI error = %.3e\n', line_err);

    ia = round(numel(theta_list)/4);

    slice_err = norm(gI_rec(:,ia) - gI_true(:,ia)) / max(norm(gI_true(:,ia)), eps);
    fprintf('Single-slice gI error at theta %.1f = %.3e\n', theta_list(ia), slice_err);

    figure('Color','w');
    subplot(2,2,1); plot(real(gI_true(:,ia))); title('real gI true');
    subplot(2,2,2); plot(real(gI_rec(:,ia)));  title('real gI rec');
    subplot(2,2,3); plot(imag(gI_true(:,ia))); title('imag gI true');
    subplot(2,2,4); plot(imag(gI_rec(:,ia)));  title('imag gI rec');

    figure('Color','w');
    subplot(2,2,1); imagesc(real(gI_true)); axis image; colorbar; title('Re[g_{\theta}] truth');
    subplot(2,2,2); imagesc(real(gI_rec));  axis image; colorbar; title('Re[g_{\theta}] rec');
    subplot(2,2,3); imagesc(imag(gI_true)); axis image; colorbar; title('Im[g_{\theta}] truth');
    subplot(2,2,4); imagesc(imag(gI_rec));  axis image; colorbar; title('Im[g_{\theta}] rec');

else
    warning('No true line-domain gI stored. Pass ''trueImZ'', ImZ_true into recover_ImZ_by_cartesian_KK.');
end

% ---------------- SUPPORT-BASED SCALE CORRECTION ----------------

gR = ifft2c(ReZ_meas);
gI_rec = ifft2c(ImZ_rec);

% ---------------- PLOTS ----------------
if show_plots
    figure('Color','w','Position',[100 100 1500 500]);

    subplot(1,3,1);
    imagesc(supp1); axis image; colorbar;
    title('E1 support');

    subplot(1,3,2);
    imagesc(supp2); axis image; colorbar;
    title('E2 support');

    subplot(1,3,3);
    imagesc(CSg); axis image; colorbar;
    title('support of g = IFT(A1^*A2)');

    colormap parula;

    figure('Color','w','Position',[100 100 1600 800]);

    ax(1) = subplot(2,3,1);
    imagesc(ReZ_meas); axis image; colorbar;
    title('Re(A1^*A2)');

    ax(2) = subplot(2,3,2);
    imagesc(ImZ_true); axis image; colorbar;
    title('True Im(A1^*A2)');

    ax(3) = subplot(2,3,3);
    imagesc(ImZ_rec); axis image; colorbar;
    title('Raw recovered Im(A1^*A2)');

    ax(4) = subplot(2,3,4);
    imagesc(ImZ_rec); axis image; colorbar;
    title('Support-scaled recovered Im(A1^*A2)');

    ax(5) = subplot(2,3,5);
    imagesc(ImZ_rec - ImZ_true); axis image; colorbar;
    title('\Delta ImZ after support scale');

    subplot(2,3,6);
    imagesc(log10(abs(Z_true)+1e-12)); axis image; colorbar;
    title('log10 |A1^*A2|');

    colormap parula;

    linkaxes(ax);

end

% ========================================================================
% Apply DBFH
% ========================================================================

mag1 = sqrt(max(I_1, 0));

sys = diff_dbfh_prepare_system( ...
    mag1, Z_true, supp1, supp2, ...
    'lambda', 0, ...
    'mag1RelThresh', 1e-10, ...
    'cRelThresh', 0);


% --- Hutchinson diagonal ---
nsamp = 30;
blockSize = 1;
useGPU = false;

d = hutchinson_diag_AtA(sys, nsamp, blockSize, useGPU);

d = double(gather(d(:)));

dpos = d(d > 0 & isfinite(d));
dfloor = 1e-8 * median(dpos);

p = 1 ./ sqrt(max(d, dfloor));

p_med = median(p(isfinite(p) & p > 0));
p_cap = 1e2 * p_med;   % try 1e3, 1e4, 1e5

p = min(p, p_cap);

fprintf('p length/sys.n = %d / %d\n', numel(p), sys.n);
fprintf('p range = %.3e / %.3e\n', min(p), max(p));

Afun = make_Afun_lsqr_stacked_diagprecond(sys, p);

% --- Build stacked RHS ---
b = sys.b;
bstack = [real(b); imag(b)];

% --- Solve ---
tol = 1e-12;
maxit = 2000;

tic;
[xstack, flag, relres, iter, resvec] = lsqr(Afun, bstack, tol, maxit);
solve_time = toc;

x1vec = xstack(1:sys.n) + 1i*xstack(sys.n+1:end);

% --- Recover fields ---
out = diff_dbfh_unpack_solution(sys, x1vec, 'normalizePhasor', true);

X1 = out.X1;
X2 = out.X2;

A1_rec = out.A1;
A2_rec = out.A2;

E1_rec = out.E1;
E2_rec = out.E2;
E_rec  = out.E;

phi0 = angle(sum(conj(E(logical(supp12))) .* E_rec(logical(supp12))));
E_rec_aligned = E_rec .* exp(-1i*phi0);

% fprintf('leak1 = %.3e\\n', out.leak1);
% fprintf('leak2 = %.3e\\n', out.leak2);


figure;imagesc(angle(E_rec_aligned).*supp12);
figure;imagesc(angle(E).*supp12);



% ========================================================================
% SUPPORT OF g = IFT[A1^* A2]
% support is all displacement vectors x2 - x1
% ========================================================================
function CSg = correlation_support(E1_mask, E2_mask)

big = conv2(double(flipud(fliplr(E1_mask))), double(E2_mask), 'same');
CSg = big > 0;

end

% ========================================================================
% HWO HEX SEGMENTED APERTURE
% ========================================================================
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

% ========================================================================
% fft helper functions
% ========================================================================
function F = fft2c(x)
F = fftshift(fft2(ifftshift(x))) / sqrt(numel(x));
end

function x = ifft2c(F)
x = fftshift(ifft2(ifftshift(F))) * sqrt(numel(F));
end

function F = fft1c(x)
F = fftshift(fft(ifftshift(x)));
end

function x = ifft1c(F)
x = fftshift(ifft(ifftshift(F)));
end

% ========================================================================
% KK ImZ recovery
% ========================================================================
function [ImZ_rec, info] = recover_ImZ_by_cartesian_KK(ReZ, CSg, theta_list, varargin)
%RECOVER_IMZ_BY_CARTESIAN_KK
%
% Instead of:
%   R_Re = radon(ReZ)
%   gR   = ifft1c(R_Re)
%
% this uses the exact Cartesian object:
%   gR_cart = ifft2c(ReZ)
%
% and samples radial lines directly in the Cartesian correlation plane.

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

% ------------------------------------------------------------
% FFT-centered Cartesian coordinates.
% Physical convention:
%   x positive to the right
%   y positive upward
% Matrix rows increase downward, so y(row) = center - row.
% ------------------------------------------------------------
c = floor(N/2) + 1;

x = (1:N) - c;        % columns
y = c - (1:N);        % rows, positive upward

[Xcart, Ycart] = meshgrid(x, y);

% Radial coordinate sampled on the same integer grid.
s = x(:);

num_rho    = numel(s);
num_angles = numel(theta_list);

% ------------------------------------------------------------
% Exact Cartesian correlation-plane real channel.
% This is the object that radon+ifft1 was trying to approximate.
% ------------------------------------------------------------
gR_cart = ifft2c(ReZ);

[FR, FI] = make_complex_interpolants(gR_cart, x, y, interpMethod);

% Optional truth diagnostic.
have_true = ~isempty(trueImZ);
if have_true
    gI_true_cart = ifft2c(trueImZ);
    [FTR, FTI] = make_complex_interpolants(gI_true_cart, x, y, interpMethod);
    gI_true_all = zeros(num_rho, num_angles);
else
    gI_true_all = [];
end

% ------------------------------------------------------------
% Estimate support half-plane direction from CSg centroid.
% Same physical coordinate convention as above.
% ------------------------------------------------------------
[rr, cc] = find(CSg);

gx = cc - c;
gy = c  - rr;

cent = [mean(gx), mean(gy)];
n_hat = cent / max(norm(cent), eps);

% ------------------------------------------------------------
% Per-angle line sampling + one-sided KK/Hermitian fill.
% ------------------------------------------------------------
gR_all = zeros(num_rho, num_angles);
gI_all = zeros(num_rho, num_angles);
R_Re   = zeros(num_rho, num_angles);   % pseudo-projections, for debugging only
R_Im   = zeros(num_rho, num_angles);   % pseudo-projections, for debugging only
valid_frac = zeros(1, num_angles);

for ia = 1:num_angles

    theta = theta_list(ia);

    % Line in Cartesian correlation plane.
    % YSign lets you flip the angular convention if needed.
    xline = s .* cosd(theta);
    yline = YSign .* s .* sind(theta);

    % Directly sample gR_cart along this line.
    gR = FR(yline, xline) + 1i * FI(yline, xline);
    gR(isnan(gR)) = 0;

    % Direction used for deciding which side is known-zero.
    th = [cosd(theta), YSign * sind(theta)];

    side_sign = sign(dot(n_hat, th));
    if side_sign == 0
        continue;
    end

    side_sign = SideFlip * side_sign;

    zero_side    = side_sign * s < 0;
    support_side = side_sign * s > 0;

    gI = zeros(num_rho, 1);

    % On zero side:
    %   g = gR + i*gI = 0
    % therefore:
    %   gI = i*gR
    %
    % KKSign is kept because your previous code exposed it.
    gI(zero_side) = KKSign * 1i * gR(zero_side);

    % Hermitian fill:
    % if ImZ is real, then ifft2c(ImZ) is Hermitian:
    %   gI(-s) = conj(gI(s))
    for j = 1:num_rho
        if support_side(j)
            sm = -s(j);
            [~, jm] = min(abs(s - sm));
            gI(j) = conj(gI(jm));
        end
    end

    % Center point should be real for a Hermitian object.
    [~, j0] = min(abs(s));
    gI(j0) = real(gI(j0));

    gR_all(:,ia) = gR;
    gI_all(:,ia) = gI;

    % These are not MATLAB radon projections.
    % They are only the 1D Fourier transforms of the sampled lines.
    R_Re(:,ia) = real(fft1c(gR));
    R_Im(:,ia) = real(fft1c(gI));

    valid_frac(ia) = nnz(zero_side) / numel(zero_side);

    if have_true
        gIt = FTR(yline, xline) + 1i * FTI(yline, xline);
        gIt(isnan(gIt)) = 0;
        gI_true_all(:,ia) = gIt;
    end
end

% ------------------------------------------------------------
% Grid recovered radial slices gI_theta(s) back to Cartesian gI(x,y).
% ------------------------------------------------------------
[ThetaGrid, SGrid] = meshgrid(deg2rad(theta_list), s);

Xp = SGrid .* cos(ThetaGrid);
Yp = YSign .* SGrid .* sin(ThetaGrid);

gI_cart_real = griddata(Xp(:), Yp(:), real(gI_all(:)), Xcart, Ycart, interpMethod);
gI_cart_imag = griddata(Xp(:), Yp(:), imag(gI_all(:)), Xcart, Ycart, interpMethod);

gI_cart = gI_cart_real + 1i * gI_cart_imag;
gI_cart(isnan(gI_cart)) = 0;

% Restrict to sampled disk.
Rmax = max(abs(s));
valid_disk = sqrt(Xcart.^2 + Ycart.^2) <= Rmax;
gI_cart(~valid_disk) = 0;

% Recover ImZ from:
%   gI = ifft2c(ImZ)
ImZ_complex = fft2c(gI_cart);
ImZ_rec = real(ImZ_complex);

% ------------------------------------------------------------
% Diagnostics.
% ------------------------------------------------------------
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
% griddedInterpolant expects coordinates ordered by matrix dimension:
%   first coordinate  -> row/y
%   second coordinate -> col/x
%
% y in the main function is descending because matrix row 1 is top.
% Flip vertically so y becomes ascending.

x = x(:).';
y = y(:).';

y_asc = fliplr(y);
V_asc = flipud(V);

FR = griddedInterpolant({y_asc, x}, real(V_asc), interpMethod, 'none');
FI = griddedInterpolant({y_asc, x}, imag(V_asc), interpMethod, 'none');

end