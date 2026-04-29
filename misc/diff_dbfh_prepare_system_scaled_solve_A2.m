function sys = diff_dbfh_prepare_system_scaled_solve_A2(mag1, Z, supp1, supp2, opts)
%DIFF_DBFH_PREPARE_SYSTEM_SCALED_SOLVE_A2
% Augmented differential-DBFH system that solves for X1 and E2 directly.
%
% This avoids the fragile eliminated equation
%       A2 = (Z ./ |A1|) .* X1
% and instead enforces the linear cross-term equation
%       Z .* X1 - |A1| .* A2 = 0,
% where
%       A2 = fft2c(E2),   E2 is nonzero only on supp2.
%
% Unknown vector:
%       x = [x1; e2]
%
%   x1 : X1 on focal mask K, where A1 = |A1| X1
%   e2 : complex pupil-plane field E2 only on supp2
%
% Equations:
%   1. ifft2c(|A1| .* X1) = 0 outside supp1
%   2. Z .* X1 - |A1| .* fft2c(E2) = 0 on crossMask, default K
%   3. X1(anchor) = anchorVal
%   4. optional Tikhonov rows on x1 and/or e2
%
% IMPORTANT CONVENTION
%   This function uses CENTERED UNITARY transforms internally:
%       fft2c(x)  = fftshift(fft2(ifftshift(x))) / sqrt(numel(x))
%       ifft2c(F) = fftshift(ifft2(ifftshift(F))) * sqrt(numel(F))
%
% Required inputs:
%   mag1  : HxW real nonnegative |A1| = sqrt(I1), centered focal grid
%   Z     : HxW complex cross term A1^* A2, centered focal grid
%   supp1 : HxW logical pupil support for E1
%   supp2 : HxW logical pupil support for E2
%
% Main options:
%   lambda              default 0       common Tikhonov fallback
%   lambdaX1            default NaN     if NaN, uses lambda
%   lambdaE2            default NaN     if NaN, uses lambda
%   anchor              default []      linear index inside K
%   anchorVal           default 1+0i
%   mag1RelThresh       default 1e-12   K threshold on |A1|/max(|A1|)
%   mag1AbsThresh       default 0
%   cRelThresh          default 0       optional threshold on |Z|
%   cAbsThresh          default 0
%   crossWeight         default 1       scalar row weight for cross equation
%   supportScale        default 1       scalar row weight for support equation
%   anchorWeight        default 1
%   crossMask           default []      custom HxW logical mask; otherwise K
%   anchorStrategy      default 'maxMag1'; 'maxMag1', 'median', 'maxCombined'
%
% Output:
%   sys.Amul, sys.ATmul, sys.b, sys.m, sys.n are compatible with the same
%   LSQR/Hutchinson machinery used for the previous DBFH operators.
%
% Use diff_dbfh_unpack_solution_solve_A2.m to unpack the solution.

    arguments
        mag1 double
        Z double
        supp1 logical
        supp2 logical
        opts.lambda (1,1) double {mustBeNonnegative} = 0
        opts.lambdaX1 (1,1) double = NaN
        opts.lambdaE2 (1,1) double = NaN
        opts.anchor = []
        opts.anchorVal (1,1) double = 1+0i
        opts.mag1RelThresh (1,1) double {mustBeNonnegative} = 1e-12
        opts.mag1AbsThresh (1,1) double {mustBeNonnegative} = 0
        opts.cRelThresh (1,1) double {mustBeNonnegative} = 0
        opts.cAbsThresh (1,1) double {mustBeNonnegative} = 0
        opts.crossWeight (1,1) double {mustBePositive} = 1
        opts.supportScale (1,1) double {mustBePositive} = 1
        opts.anchorWeight (1,1) double {mustBePositive} = 1
        opts.crossMask = []
        opts.anchorStrategy char = 'maxMag1'
        opts.warnOnSupportOverlap (1,1) logical = true
    end

    [H,W] = size(mag1);
    assert(isequal(size(mag1), size(Z), size(supp1), size(supp2)), ...
        'mag1, Z, supp1, and supp2 must be HxW and of equal size.');
    assert(isreal(mag1), 'mag1 must be real.');
    assert(all(mag1(:) >= 0 | isnan(mag1(:))), 'mag1 must be nonnegative.');

    if opts.warnOnSupportOverlap && any(supp1(:) & supp2(:))
        warning('diff_dbfh_prepare_system_solve_A2:SupportOverlap', ...
            'supp1 and supp2 overlap. Verify this is intentional.');
    end

    lambdaX1 = opts.lambda;
    lambdaE2 = opts.lambda;
    if ~isnan(opts.lambdaX1), lambdaX1 = opts.lambdaX1; end
    if ~isnan(opts.lambdaE2), lambdaE2 = opts.lambdaE2; end
    assert(lambdaX1 >= 0 && lambdaE2 >= 0, 'lambdaX1/lambdaE2 must be nonnegative.');

    finiteZ = isfinite(real(Z)) & isfinite(imag(Z));
    mag1Max = max(mag1(isfinite(mag1)), [], 'all');
    if isempty(mag1Max) || mag1Max <= 0
        error('mag1 has no positive finite pixels.');
    end

    absZ = abs(Z);
    zMax = max(absZ(finiteZ), [], 'all');
    if isempty(zMax) || ~isfinite(zMax)
        zMax = 0;
    end

    mag1Floor = max(opts.mag1AbsThresh, opts.mag1RelThresh * mag1Max);
    zFloor    = max(opts.cAbsThresh,    opts.cRelThresh    * zMax);

    % X1 is only meaningful where |A1| is nonzero enough and Z is finite.
    K = isfinite(mag1) & finiteZ & (mag1 > mag1Floor);
    if zFloor > 0
        K = K & (absZ > zFloor);
    end

    nX = nnz(K);
    if nX == 0
        error('No valid focal pixels in K. Relax thresholds or check mag1/Z.');
    end

    idxE2 = find(supp2);
    nE = numel(idxE2);
    if nE == 0
        error('supp2 is empty; there is no E2 field to solve.');
    end

    if isempty(opts.crossMask)
        crossMask = K;
    else
        assert(isequal(size(opts.crossMask), [H,W]) && islogical(opts.crossMask), ...
            'crossMask must be an HxW logical array.');
        crossMask = opts.crossMask & K & finiteZ & isfinite(mag1);
    end
    idxCross = find(crossMask);
    nCross = numel(idxCross);
    if nCross == 0
        error('crossMask is empty after intersection with K.');
    end

    % W1 is |A1| on K, zero elsewhere. ZK is Z on crossMask/K.
    W1 = zeros(H,W);
    W1(K) = mag1(K);

    % Anchor fixes the homogeneous scale/phase ambiguity.
    if isempty(opts.anchor)
        anchor_idx = choose_anchor(K, mag1, Z, opts.anchorStrategy);
    else
        anchor_idx = opts.anchor;
        assert(isscalar(anchor_idx) && anchor_idx >= 1 && anchor_idx <= H*W, ...
            'opts.anchor must be a valid linear index.');
        assert(K(anchor_idx), 'opts.anchor must lie inside K.');
    end
    anchor_mask = false(H,W);
    anchor_mask(anchor_idx) = true;

    idx1_out = find(~supp1);
    n1 = numel(idx1_out);

    m = n1 + nCross + 1 + (lambdaX1 > 0)*nX + (lambdaE2 > 0)*nE;
    n = nX + nE;

    toFullX1 = @(x1vec) embed_on_mask(x1vec, K, H, W);
    toVecX1  = @(X) X(K);
    toFullE2 = @(e2vec) embed_on_mask(e2vec, supp2, H, W);
    toVecE2  = @(E) E(supp2);

    Amul = @(xvec) local_A_times(xvec, nX, nE, H, W, ...
        idx1_out, idxCross, W1, mag1, Z, anchor_mask, ...
        opts.supportScale, opts.crossWeight, opts.anchorWeight, ...
        lambdaX1, lambdaE2, toFullX1, toFullE2);

    ATmul = @(yvec) local_AH_times(yvec, nX, nE, H, W, ...
        idx1_out, idxCross, W1, mag1, Z, anchor_mask, ...
        opts.supportScale, opts.crossWeight, opts.anchorWeight, ...
        lambdaX1, lambdaE2, toVecX1, toVecE2);

    b = zeros(m,1);
    b(n1+nCross+1) = opts.anchorWeight * opts.anchorVal;

    sys = struct();
    sys.Amul = Amul;
    sys.ATmul = ATmul;
    sys.b = b;
    sys.m = m;
    sys.n = n;
    sys.nX1 = nX;
    sys.nE2 = nE;
    sys.K = K;
    sys.crossMask = crossMask;
    sys.idxCross = idxCross;
    sys.idx1_out = idx1_out;
    sys.idxE2 = idxE2;
    sys.H = H;
    sys.W = W;
    sys.mag1 = mag1;
    sys.Z = Z;
    sys.C = Z;          % compatibility alias
    sys.W1 = W1;
    sys.supp1 = supp1;
    sys.supp2 = supp2;
    sys.anchor_idx = anchor_idx;
    sys.anchorVal = opts.anchorVal;
    sys.mag1Floor = mag1Floor;
    sys.cFloor = zFloor;
    sys.lambdaX1 = lambdaX1;
    sys.lambdaE2 = lambdaE2;
    sys.supportScale = opts.supportScale;
    sys.crossWeight = opts.crossWeight;
    sys.anchorWeight = opts.anchorWeight;
    sys.toFullX1 = toFullX1;
    sys.toVecX1 = toVecX1;
    sys.toFullE2 = toFullE2;
    sys.toVecE2 = toVecE2;
    sys.fftConvention = 'centered-unitary';
    sys.description = 'Augmented differential DBFH: unknowns are X1 on K and E2 on supp2; no Z/|A1| division.';
end

% =====================================================================
% Forward operator
% =====================================================================
function y = local_A_times(xvec, nX, nE, H, W, idx1_out, idxCross, ...
    W1, mag1, Z, anchor_mask, supportScale, crossWeight, anchorWeight, ...
    lambdaX1, lambdaE2, toFullX1, toFullE2)

    assert(numel(xvec) == nX+nE, 'xvec length must equal sys.n.');

    x1vec = xvec(1:nX);
    e2vec = xvec(nX+1:nX+nE);

    X1 = toFullX1(x1vec);
    E2 = toFullE2(e2vec);

    % Block 1: E1 compact support.
    E1 = ifft2c_unitary(W1 .* X1);
    y1 = supportScale * E1(idx1_out);

    % Block 2: cross-term consistency, no division by |A1|.
    A2 = fft2c_unitary(E2);
    R = Z .* X1 - mag1 .* A2;
    y2 = crossWeight * R(idxCross);

    % Block 3: anchor.
    yAnchor = anchorWeight * sum(X1(anchor_mask), 'all');

    y = [y1; y2; yAnchor];

    if lambdaX1 > 0
        y = [y; lambdaX1 * x1vec];
    end
    if lambdaE2 > 0
        y = [y; lambdaE2 * e2vec];
    end
end

% =====================================================================
% Adjoint operator
% =====================================================================
function x = local_AH_times(yvec, nX, nE, H, W, idx1_out, idxCross, ...
    W1, mag1, Z, anchor_mask, supportScale, crossWeight, anchorWeight, ...
    lambdaX1, lambdaE2, toVecX1, toVecE2)

    n1 = numel(idx1_out);
    nCross = numel(idxCross);

    ptr = 0;
    y1 = yvec(ptr+1:ptr+n1); ptr = ptr+n1;
    y2 = yvec(ptr+1:ptr+nCross); ptr = ptr+nCross;
    yAnchor = yvec(ptr+1); ptr = ptr+1;

    if lambdaX1 > 0
        yLamX = yvec(ptr+1:ptr+nX); ptr = ptr+nX;
    else
        yLamX = [];
    end

    if lambdaE2 > 0
        yLamE = yvec(ptr+1:ptr+nE); %#ok<NASGU>
    else
        yLamE = [];
    end

    % Adjoint of block 1: supportScale * ifft2c_unitary(W1 .* X1).
    E1out = zeros(H,W, 'like', yvec);
    E1out(idx1_out) = y1;
    gX1_1 = supportScale * conj(W1) .* fft2c_unitary(E1out);

    % Adjoint of block 2:
    % y2 = crossWeight * (Z .* X1 - mag1 .* fft2c(E2)) on idxCross.
    Y = zeros(H,W, 'like', yvec);
    Y(idxCross) = y2;

    gX1_2 = crossWeight * conj(Z) .* Y;

    gA2 = -crossWeight * mag1 .* Y;
    gE2_full = ifft2c_unitary(gA2);  % adjoint of fft2c_unitary

    % Adjoint of anchor.
    gX1_anchor = zeros(H,W, 'like', yvec);
    gX1_anchor(anchor_mask) = anchorWeight * yAnchor;

    gX1_full = gX1_1 + gX1_2 + gX1_anchor;
    gE2_vec = toVecE2(gE2_full);

    gX1_vec = toVecX1(gX1_full);

    if lambdaX1 > 0
        gX1_vec = gX1_vec + lambdaX1 * yLamX;
    end
    if lambdaE2 > 0
        % yLamE was parsed only when lambdaE2 > 0.
        nBase = n1 + nCross + 1 + (lambdaX1 > 0)*nX;
        yLamE = yvec(nBase+1:nBase+nE);
        gE2_vec = gE2_vec + lambdaE2 * yLamE;
    end

    x = [gX1_vec; gE2_vec];
end

% =====================================================================
% Helpers
% =====================================================================
function X = embed_on_mask(x, mask, H, W)
    X = zeros(H,W, 'like', x);
    X(mask) = x;
end

function anchor_idx = choose_anchor(K, mag1, Z, anchorStrategy)
    idxK = find(K);
    strategy = lower(anchorStrategy);

    switch strategy
        case 'median'
            [H,W] = size(K);
            [rK,cK] = ind2sub([H,W], idxK);
            [~,mid] = min((rK - median(rK)).^2 + (cK - median(cK)).^2);
            anchor_idx = idxK(mid);

        case 'maxmag1'
            score = mag1(idxK);
            [~,ii] = max(score);
            anchor_idx = idxK(ii);

        case 'maxcombined'
            q = abs(Z);
            qMax = max(q(idxK));
            if qMax > 0
                score = mag1(idxK) .* (q(idxK) / qMax);
            else
                score = mag1(idxK);
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
