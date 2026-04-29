
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
