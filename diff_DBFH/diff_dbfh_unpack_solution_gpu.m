function out = diff_dbfh_unpack_solution_gpu(sys, xvec, varargin)
%DIFF_DBFH_UNPACK_SOLUTION_GPU
% GPU-compatible unpacker for diff_dbfh_prepare_system_hardanchor_gpu.
% If sys/xvec are GPU-resident, returned arrays remain on GPU.

p = inputParser;
p.addParameter('normalizeX1', false, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
normalizeX1 = p.Results.normalizeX1;

assert(isfield(sys,'hardAnchor') && sys.hardAnchor, ...
    'sys must come from a hard-anchor solve-A2 system.');
assert(numel(xvec) == sys.n, 'xvec must have sys.n elements.');

nX = sys.nX1;
nE = sys.nE2;
H = sys.H;
W = sys.W;

x1vec = xvec(1:nX);
e2vec = xvec(nX+1:nX+nE);

X1_raw = zeros(H,W,'like',xvec);
X1_raw(sys.Kfree) = x1vec(:);
X1_raw(sys.anchor_idx) = sys.anchorVal;

if normalizeX1
    amp = abs(X1_raw);
    X1 = zeros(H,W,'like',xvec);
    valid = sys.K & amp > 0;
    X1(valid) = X1_raw(valid) ./ amp(valid);
else
    X1 = X1_raw;
end

E2 = zeros(H,W,'like',xvec);
E2(sys.supp2) = e2vec(:);

A1 = sys.mag1 .* X1;
E1 = ifft2c_unitary(A1);

A2 = fft2c_unitary(E2);
E  = E1 + E2;

cross_res = sys.Z .* X1 - sys.mag1 .* A2;

out = struct();
out.X1 = X1;
out.X1_raw = X1_raw;
out.E2 = E2;
out.A1 = A1;
out.A2 = A2;
out.E1 = E1;
out.E = E;
out.cross_residual = cross_res;
out.cross_relres = norm(cross_res(sys.crossMask)) / max(norm(sys.Z(sys.crossMask) .* X1(sys.crossMask)), eps);
out.leak1 = norm(E1(~sys.supp1)) / max(norm(E1(:)), eps);
out.leak2 = norm(E2(~sys.supp2)) / max(norm(E2(:)), eps);
out.raw_phasor_abs_median = median(abs(X1_raw(sys.K)));
out.raw_phasor_abs_mad = median(abs(abs(X1_raw(sys.K)) - out.raw_phasor_abs_median));
end

function F = fft2c_unitary(x)
F = fftshift(fft2(ifftshift(x))) / sqrt(numel(x));
end

function x = ifft2c_unitary(F)
x = fftshift(ifft2(ifftshift(F))) * sqrt(numel(F));
end
