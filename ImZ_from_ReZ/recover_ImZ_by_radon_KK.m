function [ImZ_rec, info] = recover_ImZ_by_radon_KK(ReZ, CSg, theta_list, varargin)

p = inputParser;
p.addParameter('KKSign', +1, @(x) isnumeric(x) && isscalar(x) && any(x == [-1,+1]));
p.addParameter('SideFlip', +1, @(x) isnumeric(x) && isscalar(x) && any(x == [-1,+1]));
p.addParameter('InterpMethod', 'linear', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});

KKSign  = p.Results.KKSign;
SideFlip = p.Results.SideFlip;
interpMethod = char(p.Results.InterpMethod);

N = size(ReZ,1);
assert(size(ReZ,1) == size(ReZ,2), 'Use square grids for this prototype.');

R_Re = radon(ReZ, theta_list);
[num_rho, num_angles] = size(R_Re);

R_Im = zeros(size(R_Re));
gR_all = zeros(num_rho, num_angles);
gI_all = zeros(num_rho, num_angles);
valid_frac = zeros(1,num_angles);

% estimate half-plane normal from CSg centroid
[rr,cc] = find(CSg);
cx = (N+1)/2;
cy = (N+1)/2;

gx = cc - cx;
gy = cy - rr;

cent = [mean(gx), mean(gy)];
n_hat = cent / max(norm(cent), eps);

s = ((1:num_rho).' - (num_rho+1)/2);

for ia = 1:num_angles

    theta = theta_list(ia);

    % MATLAB radon convention: projection coordinate approximately
    % x*cos(theta) + y*sin(theta)
    th = [cosd(theta), sind(theta)];

    side_sign = sign(dot(n_hat, th));
    if side_sign == 0
        continue;
    end

    side_sign = SideFlip * side_sign;

    proj_Re = R_Re(:,ia);

    % Projection-slice:
    % IFT1[ R_theta[Re z] ] = g_R,theta
    gR = ifft1c(proj_Re);

    zero_side    = side_sign * s < 0;
    support_side = side_sign * s > 0;

    gI = zeros(num_rho,1);

    % On zero side: gR + i*gI = 0.
    gI(zero_side) = KKSign * 1i * gR(zero_side);

    % Hermitian fill
    for j = 1:num_rho
        if support_side(j)
            sm = -s(j);
            [~,jm] = min(abs(s - sm));
            gI(j) = conj(gI(jm));
        end
    end

    % boundary point
    [~,j0] = min(abs(s));
    gI(j0) = 0;

    gR_all(:,ia) = gR;
    gI_all(:,ia) = gI;

    proj_Im = fft1c(gI);
    R_Im(:,ia) = real(proj_Im);

    valid_frac(ia) = nnz(zero_side) / numel(zero_side);
end

% ------------------------------------------------------------
% NEW PATH:
% interpolate recovered radial slices gI_theta(s) to Cartesian gI(x,y),
% then FFT2 to get ImZ.
% ------------------------------------------------------------

% polar samples of gI in correlation/pupil plane
[ThetaGrid, SGrid] = meshgrid(deg2rad(theta_list), s);

Xp = SGrid .* cos(ThetaGrid);
Yp = SGrid .* sin(ThetaGrid);

% Cartesian correlation-plane grid matching fft2c convention
[xg, yg] = meshgrid((-N/2):(N/2-1), (-N/2):(N/2-1));

% interpolate real and imaginary parts separately
gI_cart_real = griddata(Xp(:), Yp(:), real(gI_all(:)), xg, yg, interpMethod);
gI_cart_imag = griddata(Xp(:), Yp(:), imag(gI_all(:)), xg, yg, interpMethod);

gI_cart = gI_cart_real + 1i*gI_cart_imag;
gI_cart(isnan(gI_cart)) = 0;

% restrict to sampled disk
Rmax = max(abs(s));
valid_disk = sqrt(xg.^2 + yg.^2) <= Rmax;
gI_cart(~valid_disk) = 0;

% recover ImZ from gI = IFT2[ImZ]
ImZ_complex = fft2c(gI_cart);

% ImZ should be real-valued
ImZ_rec = real(ImZ_complex);

info = struct();
info.R_Re = R_Re;
info.R_Im = R_Im;
info.gR_all = gR_all;
info.gI_all = gI_all;
info.gI_cart = gI_cart;
info.ImZ_complex = ImZ_complex;
info.s = s;
info.valid_disk = valid_disk;
info.valid_angle_fraction = valid_frac;
info.KKSign = KKSign;
info.SideFlip = SideFlip;
info.n_hat = n_hat;
info.interpMethod = interpMethod;

end


function F = fft2c(x)
F = fftshift(fft2(ifftshift(x))) / sqrt(numel(x));
end

function x = ifft1c(F)
x = fftshift(ifft(ifftshift(F)));
end

function F = fft1c(x)
F = fftshift(fft(ifftshift(x)));
end