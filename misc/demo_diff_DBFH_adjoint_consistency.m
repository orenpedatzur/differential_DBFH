function demo_diff_DBFH_adjoint_consistency()
%DEMO_DIFF_DBFH_ADJOINT_CONSISTENCY
% Check <A x, y> = <x, A^H y> for the operator used in diff_DBFH.

rng(1);

% -----------------------------
% problem size
% -----------------------------
Ny = 128;
Nx = 128;
N  = Ny * Nx;

% -----------------------------
% synthetic data
% -----------------------------
I_full = abs(randn(Ny, Nx)).^2 + 0.1;
sqrtI  = sqrt(I_full);

C = randn(Ny, Nx) + 1i * randn(Ny, Nx);

% compact supports in pupil plane
[xg, yg] = meshgrid(1:Nx, 1:Ny);
cx = (Nx + 1)/2;
cy = (Ny + 1)/2;

rE  = min(Nx, Ny) * 0.22;
rdE = min(Nx, Ny) * 0.06;

cs_E  = ((xg - cx).^2 + (yg - cy).^2) <= rE^2;
cs_dE = ((xg - (cx + 10)).^2 + (yg - (cy - 6)).^2) <= rdE^2;

m = double(cs_dE);

mask_out_E  = ~cs_E;
mask_out_dE = ~cs_dE;

lambdaE  = 1.3;
lambdaDE = 0.7;

% row count of stacked operator
n1 = N;
n2 = nnz(mask_out_E);
n3 = nnz(mask_out_dE);
Mtot = n1 + n2 + n3;

% -----------------------------
% random test vectors
% -----------------------------
x = randn(N,1) + 1i * randn(N,1);
y = randn(Mtot,1) + 1i * randn(Mtot,1);

% -----------------------------
% apply forward and adjoint
% -----------------------------
Ax  = diff_dbfh_operator(x, 'notransp', ...
    sqrtI, C, m, mask_out_E, mask_out_dE, lambdaE, lambdaDE, Ny, Nx);

AHy = diff_dbfh_operator(y, 'transp', ...
    sqrtI, C, m, mask_out_E, mask_out_dE, lambdaE, lambdaDE, Ny, Nx);

% -----------------------------
% inner product test
% MATLAB inner product uses conj(a(:)).'*b(:)
% -----------------------------
lhs = Ax(:)' * y(:);
rhs = x(:)' * AHy(:);

abs_err = abs(lhs - rhs);
rel_err = abs_err / max([abs(lhs), abs(rhs), eps]);

fprintf('Adjoint consistency test\n');
fprintf('  <A x, y>          = %.16e %+.16ei\n', real(lhs), imag(lhs));
fprintf('  <x, A^H y>        = %.16e %+.16ei\n', real(rhs), imag(rhs));
fprintf('  absolute error    = %.3e\n', abs_err);
fprintf('  relative error    = %.3e\n', rel_err);

if rel_err < 1e-10
    fprintf('PASS\n');
else
    fprintf('FAIL\n');
end

end


function y = diff_dbfh_operator(x, transp_flag, sqrtI, C, m, mask_out_E, mask_out_dE, lambdaE, lambdaDE, Ny, Nx)
% Operator for LSQR:
%
% forward:
%   y = [ r1(:) ;
%         lambdaE  * E(outside cs_E) ;
%         lambdaDE * deltaE(outside cs_dE) ]
%
% where
%   A      = sqrtI .* X
%   E      = ifft2c(A)
%   deltaE = m .* E
%   deltaA = fft2c(deltaE)
%   r1     = C .* X - sqrtI .* deltaA

if strcmp(transp_flag, 'notransp')

    X = reshape(x, Ny, Nx);

    A = sqrtI .* X;
    E = ifft2c(A);
    deltaE = m .* E;
    deltaA = fft2c(deltaE);

    r1 = C .* X - sqrtI .* deltaA;
    r2 = lambdaE  * E(mask_out_E);
    r3 = lambdaDE * deltaE(mask_out_dE);

    y = [r1(:); r2; r3];

else
    idx1 = Ny * Nx;
    n2 = nnz(mask_out_E);
    n3 = nnz(mask_out_dE);

    r1  = reshape(x(1:idx1), Ny, Nx);
    r2v = x(idx1+1 : idx1+n2);
    r3v = x(idx1+n2+1 : idx1+n2+n3);

    % block 1 adjoint
    term1 = conj(C) .* r1;

    tmp = sqrtI .* r1;
    tmp = ifft2c(tmp);
    tmp = m .* tmp;
    tmp = fft2c(tmp);
    term2 = sqrtI .* tmp;

    g1 = term1 - term2;

    % block 2 adjoint
    R2 = zeros(Ny, Nx);
    R2(mask_out_E) = lambdaE * r2v;
    g2 = sqrtI .* fft2c(R2);

    % block 3 adjoint
    R3 = zeros(Ny, Nx);
    R3(mask_out_dE) = lambdaDE * r3v;
    g3 = sqrtI .* fft2c(m .* R3);

    y = g1(:) + g2(:) + g3(:);
end

end


function F = fft2c(x)
F = fftshift(fft2(ifftshift(x))) / sqrt(numel(x));
end

function x = ifft2c(F)
x = fftshift(ifft2(ifftshift(F))) * sqrt(numel(F));
end