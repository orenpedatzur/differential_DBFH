function sys = diff_dbfh_prepare_system(mag1, Z, supp1, supp2, opts)
%DIFF_DBFH_PREPARE_SYSTEM
% Differential DBFH system that solves for X1 and E2 directly, with
% X1(anchor) fixed exactly by eliminating that unknown.
%
% Unknown vector:
%       x = [x1_free; e2]
%
%   x1_free : X1 on K, except the anchor pixel
%   e2      : complex pupil-plane field E2 only on supp2
%
% Equations:
%   1. ifft2c(|A1| .* X1) = 0 outside supp1
%   2. Z .* X1 - |A1| .* fft2c(E2) = 0 on crossMask, default K
%   3. optional Tikhonov rows on x1_free and/or e2
%
% There is NO anchor row. Instead X1(anchor) = anchorVal is inserted as a
% fixed value and its contribution is moved to the RHS. This prevents LSQR
% from escaping to the near-zero solution by sacrificing a soft anchor row.
%
% Uses centered unitary FFTs internally.

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
        warning('diff_dbfh_prepare_system_solve_A2_hardanchor:SupportOverlap', ...
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

    Kfull = isfinite(mag1) & finiteZ & (mag1 > mag1Floor);
    if zFloor > 0
        Kfull = Kfull & (absZ > zFloor);
    end

    nK = nnz(Kfull);
    if nK == 0
        error('No valid focal pixels in K. Relax thresholds or check mag1/Z.');
    end

    % Choose the fixed anchor inside the full valid focal mask.
    if isempty(opts.anchor)
        anchor_idx = choose_anchor(Kfull, mag1, Z, opts.anchorStrategy);
    else
        anchor_idx = opts.anchor;
        assert(isscalar(anchor_idx) && anchor_idx >= 1 && anchor_idx <= H*W, ...
            'opts.anchor must be a valid linear index.');
        assert(Kfull(anchor_idx), 'opts.anchor must lie inside K.');
    end

    anchor_mask = false(H,W);
    anchor_mask(anchor_idx) = true;

    Kfree = Kfull;
    Kfree(anchor_idx) = false;

    nX = nnz(Kfree);
    idxE2 = find(supp2);
    nE = numel(idxE2);
    if nE == 0
        error('supp2 is empty; there is no E2 field to solve.');
    end

    if isempty(opts.crossMask)
        crossMask = Kfull;
    else
        assert(isequal(size(opts.crossMask), [H,W]) && islogical(opts.crossMask), ...
            'crossMask must be an HxW logical array.');
        crossMask = opts.crossMask & Kfull & finiteZ & isfinite(mag1);
    end
    idxCross = find(crossMask);
    nCross = numel(idxCross);
    if nCross == 0
        error('crossMask is empty after intersection with K.');
    end

    W1 = zeros(H,W);
    W1(Kfull) = mag1(Kfull);

    idx1_out = find(~supp1);
    n1 = numel(idx1_out);

    % Fixed anchor field and its contribution to the two homogeneous blocks.
    Xfix = zeros(H,W);
    Xfix(anchor_idx) = opts.anchorVal;

    E1fix = ifft2c(W1 .* Xfix);
    b1 = -opts.supportScale * E1fix(idx1_out);

    Rfix = Z .* Xfix;   % E2 fixed part is zero.
    b2 = -opts.crossWeight * Rfix(idxCross);

    m = n1 + nCross + (lambdaX1 > 0)*nX + (lambdaE2 > 0)*nE;
    n = nX + nE;

    toFullX1Free = @(x1vec) embed_on_mask(x1vec, Kfree, H, W);
    toVecX1Free  = @(X) X(Kfree);
    toFullE2     = @(e2vec) embed_on_mask(e2vec, supp2, H, W);
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
        b = [b; zeros(nX,1)];
    end
    if lambdaE2 > 0
        b = [b; zeros(nE,1)];
    end

    sys = struct();
    sys.Amul = Amul;
    sys.ATmul = ATmul;
    sys.b = b;
    sys.m = m;
    sys.n = n;
    sys.nX1 = nX;       % free X1 unknowns only
    sys.nE2 = nE;
    sys.nK = nK;        % full valid X1 pixels, including anchor
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
    sys.description = 'Differential DBFH solve-A2 with hard fixed X1(anchor); unknowns are X1 on Kfree and E2 on supp2.';
end

% =====================================================================
% Forward operator: linear part only. Fixed anchor is accounted for in b.
% =====================================================================
function y = local_A_times(xvec, nX, nE, H, W, idx1_out, idxCross, ...
    W1, mag1, Z, supportScale, crossWeight, ...
    lambdaX1, lambdaE2, toFullX1Free, toFullE2)

    assert(numel(xvec) == nX+nE, 'xvec length must equal sys.n.');

    x1vec = xvec(1:nX);
    e2vec = xvec(nX+1:nX+nE);

    X1free = toFullX1Free(x1vec);
    E2 = toFullE2(e2vec);

    E1 = ifft2c(W1 .* X1free);
    y1 = supportScale * E1(idx1_out);

    A2 = fft2c(E2);
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

% =====================================================================
% Adjoint operator of the linear part.
% =====================================================================
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

    E1out = zeros(H,W, 'like', yvec);
    E1out(idx1_out) = y1;
    gX1_1 = supportScale * conj(W1) .* fft2c(E1out);

    Y = zeros(H,W, 'like', yvec);
    Y(idxCross) = y2;

    gX1_2 = crossWeight * conj(Z) .* Y;

    gA2 = -crossWeight * mag1 .* Y;
    gE2_full = ifft2c(gA2);

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
