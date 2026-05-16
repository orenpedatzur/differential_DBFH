function sys = diff_dbfh_prepare_system_gpu(mag1, Z, supp1, supp2, varargin)
%DIFF_DBFH_PREPARE_SYSTEM_GPU
% GPU-compatible differential DBFH system that solves for X1 and E2 directly,
% with X1(anchor) fixed exactly by removing that pixel from the unknown vector.
%
% Unknown vector:
%   x = [x1_free ; e2]
%       x1_free : X1 on K except the hard anchor pixel
%       e2      : complex pupil field E2 on supp2
%
% Equations:
%   supportScale * ifft2c(|A1| .* X1_free) = - fixed-anchor contribution outside supp1
%   crossWeight  * (Z .* X1_free - |A1| .* fft2c(E2)) = - fixed-anchor contribution on crossMask
%   optional lambdaX1/lambdaE2 Tikhonov rows
%
% This version is GPU-clean: if mag1/Z/support masks are gpuArray, sys.Amul,
% sys.ATmul, and sys.b stay on GPU. No gather/double calls appear in the operator.

p = inputParser;
p.addParameter('lambda', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('lambdaX1', NaN, @(x) isnumeric(x) && isscalar(x));
p.addParameter('lambdaE2', NaN, @(x) isnumeric(x) && isscalar(x));
p.addParameter('anchor', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
p.addParameter('anchorVal', 1+0i, @(x) isnumeric(x) && isscalar(x));
p.addParameter('mag1RelThresh', 1e-12, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('mag1AbsThresh', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('cRelThresh', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('cAbsThresh', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('crossWeight', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('supportScale', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('crossMask', [], @(x) isempty(x) || islogical(x));
p.addParameter('anchorStrategy', 'maxMag1', @(x) ischar(x) || isstring(x));
p.addParameter('warnOnSupportOverlap', true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
opts = p.Results;

[H,W] = size(mag1);
assert(isequal(size(mag1), size(Z), size(supp1), size(supp2)), ...
    'mag1, Z, supp1, and supp2 must be HxW and of equal size.');
assert(isreal(mag1), 'mag1 must be real.');
assert(gather(all(mag1(:) >= 0 | isnan(mag1(:)))), 'mag1 must be nonnegative.');

if opts.warnOnSupportOverlap && gather(any(supp1(:) & supp2(:)))
    warning('diff_dbfh_prepare_system_hardanchor_gpu:SupportOverlap', ...
        'supp1 and supp2 overlap. Verify this is intentional.');
end

lambdaX1 = opts.lambda;
lambdaE2 = opts.lambda;
if ~isnan(opts.lambdaX1), lambdaX1 = opts.lambdaX1; end
if ~isnan(opts.lambdaE2), lambdaE2 = opts.lambdaE2; end
assert(lambdaX1 >= 0 && lambdaE2 >= 0, 'lambdaX1/lambdaE2 must be nonnegative.');

finiteZ = isfinite(real(Z)) & isfinite(imag(Z));
finiteMag = isfinite(mag1);
magVals = mag1(finiteMag);
mag1Max = max(magVals, [], 'all');
mag1Max_cpu = gather(mag1Max);
if isempty(mag1Max_cpu) || mag1Max_cpu <= 0
    error('mag1 has no positive finite pixels.');
end

absZ = abs(Z);
zVals = absZ(finiteZ);
if isempty(gather(zVals))
    zMax_cpu = 0;
else
    zMax_cpu = gather(max(zVals, [], 'all'));
    if ~isfinite(zMax_cpu), zMax_cpu = 0; end
end

mag1Floor = max(opts.mag1AbsThresh, opts.mag1RelThresh * mag1Max_cpu);
zFloor    = max(opts.cAbsThresh,    opts.cRelThresh    * zMax_cpu);

Kfull = finiteMag & finiteZ & (mag1 > mag1Floor);
if zFloor > 0
    Kfull = Kfull & (absZ > zFloor);
end

nK = gather(nnz(Kfull));
if nK == 0
    error('No valid focal pixels in K. Relax thresholds or check mag1/Z.');
end

if isempty(opts.anchor)
    anchor_idx = choose_anchor_gpu(Kfull, mag1, Z, char(opts.anchorStrategy));
else
    anchor_idx = opts.anchor;
    assert(anchor_idx >= 1 && anchor_idx <= H*W, 'opts.anchor must be a valid linear index.');
    assert(gather(Kfull(anchor_idx)), 'opts.anchor must lie inside K.');
end

anchor_mask = false(H,W,'like',supp1);
anchor_mask(anchor_idx) = true;

Kfree = Kfull;
Kfree(anchor_idx) = false;

nX = gather(nnz(Kfree));
idxE2 = gather(find(supp2));
nE = numel(idxE2);
if nE == 0
    error('supp2 is empty; there is no E2 field to solve.');
end

if isempty(opts.crossMask)
    crossMask = Kfull;
else
    crossMask = opts.crossMask;
    if isa(mag1, 'gpuArray') && ~isa(crossMask, 'gpuArray')
        crossMask = gpuArray(crossMask);
    end
    assert(isequal(size(crossMask), [H,W]) && islogical(crossMask), ...
        'crossMask must be an HxW logical array.');
    crossMask = crossMask & Kfull & finiteZ & finiteMag;
end
idxCross = gather(find(crossMask));
nCross = numel(idxCross);
if nCross == 0
    error('crossMask is empty after intersection with K.');
end

W1 = zeros(H,W,'like',mag1);
W1(Kfull) = mag1(Kfull);

idx1_out = gather(find(~supp1));
n1 = numel(idx1_out);

% Fixed anchor contribution. The unknown X1_free excludes anchor_idx.
Xfix = zeros(H,W,'like',Z);
Xfix(anchor_idx) = opts.anchorVal;

E1fix = ifft2c_unitary(W1 .* Xfix);
b1 = -opts.supportScale * E1fix(idx1_out);

Rfix = Z .* Xfix;
b2 = -opts.crossWeight * Rfix(idxCross);

m = n1 + nCross + (lambdaX1 > 0)*nX + (lambdaE2 > 0)*nE;
n = nX + nE;

toFullX1Free = @(x1vec) embed_on_mask_like(x1vec, Kfree, H, W);
toVecX1Free  = @(X) X(Kfree);
toFullE2     = @(e2vec) embed_on_mask_like(e2vec, supp2, H, W);
toVecE2      = @(E) E(supp2);

Amul = @(xvec) local_A_times(xvec, nX, nE, H, W, ...
    idx1_out, idxCross, W1, mag1, Z, ...
    opts.supportScale, opts.crossWeight, ...
    lambdaX1, lambdaE2, toFullX1Free, toFullE2);

ATmul = @(yvec) local_AH_times(yvec, nX, nE, H, W, ...
    idx1_out, idxCross, W1, mag1, Z, ...
    opts.supportScale, opts.crossWeight, ...
    lambdaX1, lambdaE2, toVecX1Free, toVecE2);

b = [b1; b2];
if lambdaX1 > 0
    b = [b; zeros(nX,1,'like',b)];
end
if lambdaE2 > 0
    b = [b; zeros(nE,1,'like',b)];
end

sys = struct();
sys.Amul = Amul;
sys.ATmul = ATmul;
sys.b = b;
sys.m = m;
sys.n = n;
sys.nX1 = nX;
sys.nE2 = nE;
sys.nK = nK;
sys.K = Kfull;
sys.Kfree = Kfree;
sys.crossMask = crossMask;
sys.idxCross = idxCross;
sys.idx1_out = idx1_out;
sys.idxE2 = idxE2;
sys.H = H;
sys.W = W;
sys.mag1 = mag1;
sys.Z = Z;
sys.C = Z;
sys.W1 = W1;
sys.supp1 = supp1;
sys.supp2 = supp2;
sys.anchor_idx = anchor_idx;
sys.anchor_mask = anchor_mask;
sys.anchorVal = opts.anchorVal;
sys.hardAnchor = true;
sys.mag1Floor = mag1Floor;
sys.cFloor = zFloor;
sys.lambdaX1 = lambdaX1;
sys.lambdaE2 = lambdaE2;
sys.supportScale = opts.supportScale;
sys.crossWeight = opts.crossWeight;
sys.toFullX1Free = toFullX1Free;
sys.toVecX1Free = toVecX1Free;
sys.toFullE2 = toFullE2;
sys.toVecE2 = toVecE2;
sys.fftConvention = 'centered-unitary';
sys.description = 'GPU-compatible hard-anchor differential DBFH solve for [X1_free; E2].';
end

function y = local_A_times(xvec, nX, nE, H, W, idx1_out, idxCross, ...
    W1, mag1, Z, supportScale, crossWeight, ...
    lambdaX1, lambdaE2, toFullX1Free, toFullE2)

assert(numel(xvec) == nX+nE, 'xvec length must equal sys.n.');

x1vec = xvec(1:nX);
e2vec = xvec(nX+1:nX+nE);

X1free = toFullX1Free(x1vec);
E2 = toFullE2(e2vec);

E1 = ifft2c_unitary(W1 .* X1free);
y1 = supportScale * E1(idx1_out);

A2 = fft2c_unitary(E2);
R = Z .* X1free - mag1 .* A2;
y2 = crossWeight * R(idxCross);

y = [y1; y2];

if lambdaX1 > 0
    y = [y; lambdaX1 * x1vec];
end
if lambdaE2 > 0
    y = [y; lambdaE2 * e2vec];
end
end

function x = local_AH_times(yvec, nX, nE, H, W, idx1_out, idxCross, ...
    W1, mag1, Z, supportScale, crossWeight, ...
    lambdaX1, lambdaE2, toVecX1Free, toVecE2)

n1 = numel(idx1_out);
nCross = numel(idxCross);

ptr = 0;
y1 = yvec(ptr+1:ptr+n1); ptr = ptr+n1;
y2 = yvec(ptr+1:ptr+nCross); ptr = ptr+nCross;

if lambdaX1 > 0
    yLamX = yvec(ptr+1:ptr+nX); ptr = ptr+nX;
else
    yLamX = [];
end

if lambdaE2 > 0
    yLamE = yvec(ptr+1:ptr+nE);
else
    yLamE = [];
end

E1out = zeros(H,W,'like',yvec);
E1out(idx1_out) = y1;
gX1_1 = supportScale * conj(W1) .* fft2c_unitary(E1out);

Y = zeros(H,W,'like',yvec);
Y(idxCross) = y2;

gX1_2 = crossWeight * conj(Z) .* Y;

gA2 = -crossWeight * mag1 .* Y;
gE2_full = ifft2c_unitary(gA2);

gX1_vec = toVecX1Free(gX1_1 + gX1_2);
gE2_vec = toVecE2(gE2_full);

if lambdaX1 > 0
    gX1_vec = gX1_vec + lambdaX1 * yLamX;
end
if lambdaE2 > 0
    gE2_vec = gE2_vec + lambdaE2 * yLamE;
end

x = [gX1_vec; gE2_vec];
end

function X = embed_on_mask_like(x, mask, H, W)
X = zeros(H,W,'like',x);
X(mask) = x;
end

function anchor_idx = choose_anchor_gpu(K, mag1, Z, anchorStrategy)
idxK = gather(find(K));
strategy = lower(anchorStrategy);

switch strategy
    case 'median'
        [H,W] = size(K);
        [rK,cK] = ind2sub([H,W], idxK);
        [~,mid] = min((rK - median(rK)).^2 + (cK - median(cK)).^2);
        anchor_idx = idxK(mid);

    case 'maxmag1'
        score = gather(mag1(idxK));
        [~,ii] = max(score);
        anchor_idx = idxK(ii);

    case 'maxcombined'
        q = abs(Z);
        qK = gather(q(idxK));
        mK = gather(mag1(idxK));
        qMax = max(qK);
        if qMax > 0
            score = mK .* (qK / qMax);
        else
            score = mK;
        end
        [~,ii] = max(score);
        anchor_idx = idxK(ii);

    otherwise
        error('Unknown anchorStrategy: %s. Use maxMag1, median, or maxCombined.', anchorStrategy);
end
end

function F = fft2c_unitary(x)
F = fftshift(fft2(ifftshift(x))) / sqrt(numel(x));
end

function x = ifft2c_unitary(F)
x = fftshift(ifft2(ifftshift(F))) * sqrt(numel(F));
end
