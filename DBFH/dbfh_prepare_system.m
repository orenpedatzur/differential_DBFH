function sys = dbfh_prepare_system(I_left, I_right, Z_est, supp_left, supp_right, opts)
% DBFH_PREPARE_SYSTEM Build a basic vanilla DBFH linear system.
%
% Known quantities:
%   I_left, I_right : focal-plane intensities from the two known subapertures
%   Z_est           : supplied complex cross term conj(A_left) .* A_right
%   supp_left       : pupil support of the left subaperture
%   supp_right      : pupil support of the right subaperture
%
% Unknown vector:
%   x = [x_left_free; x_right]
%
%   x_left_free : phase phasor X_left on K_left, except one hard anchor pixel
%   x_right     : phase phasor X_right on K_right
%
% Model:
%   A_left  = |A_left|  .* X_left
%   A_right = |A_right| .* X_right
%   Z_est   = conj(A_left) .* A_right
%
% Linear equations:
%   1. ifft2c(|A_left|  .* X_left ) = 0 outside supp_left
%   2. ifft2c(|A_right| .* X_right) = 0 outside supp_right
%   3. Z_est .* X_left - (|A_left| .* |A_right|) .* X_right = 0 on crossMask
%   4. optional Tikhonov rows on x_left_free and/or x_right
%
% The global phase is fixed by eliminating one X_left unknown:
%   X_left(anchor) = anchorVal
%
% There is no soft anchor row. The fixed anchor contribution is moved to the
% RHS, matching the hard-anchor style of diff_dbfh_prepare_system.
%
% Assumes centered unitary FFT helpers fft2c/ifft2c are on the MATLAB path.

    arguments
        I_left double
        I_right double
        Z_est double
        supp_left logical
        supp_right logical
        opts.lambda (1,1) double {mustBeNonnegative} = 0
        opts.lambdaLeft (1,1) double = NaN % Tikhonov damping rows
        opts.lambdaRight (1,1) double = NaN % Tikhonov damping rows
        opts.anchor = []
        opts.anchorVal (1,1) double = 1+0i
        opts.intensityRelThresh (1,1) double {mustBeNonnegative} = 1e-12
        opts.leftRelThresh (1,1) double = NaN
        opts.rightRelThresh (1,1) double = NaN
        opts.intensityAbsThresh (1,1) double {mustBeNonnegative} = 0
        opts.leftAbsThresh (1,1) double = NaN
        opts.rightAbsThresh (1,1) double = NaN
        opts.zRelThresh (1,1) double {mustBeNonnegative} = 0
        opts.zAbsThresh (1,1) double {mustBeNonnegative} = 0
        opts.supportScale (1,1) double {mustBePositive} = 1
        opts.crossWeight (1,1) double {mustBePositive} = 1
        opts.crossMask = []
        opts.anchorStrategy char = 'maxCombined'
        opts.warnOnSupportOverlap (1,1) logical = true
    end

    [H,W] = size(I_left);
    assert(isequal(size(I_left), size(I_right), size(Z_est), size(supp_left), size(supp_right)), ...
        'I_left, I_right, Z_est, supp_left, and supp_right must be HxW and equal size.');
    assert(isreal(I_left) && isreal(I_right), 'I_left and I_right must be real.');
    assert(all(I_left(:)  >= 0 | isnan(I_left(:))),  'I_left must be nonnegative.');
    assert(all(I_right(:) >= 0 | isnan(I_right(:))), 'I_right must be nonnegative.');

    if opts.warnOnSupportOverlap && any(supp_left(:) & supp_right(:))
        warning('vanilla_dbfh_prepare_system:SupportOverlap', ...
            'supp_left and supp_right overlap. Vanilla DBFH usually expects disjoint subapertures.');
    end

    lambdaLeft = opts.lambda;
    lambdaRight = opts.lambda;
    if ~isnan(opts.lambdaLeft),  lambdaLeft  = opts.lambdaLeft;  end
    if ~isnan(opts.lambdaRight), lambdaRight = opts.lambdaRight; end
    assert(lambdaLeft >= 0 && lambdaRight >= 0, 'lambdaLeft/lambdaRight must be nonnegative.');

    leftRel = opts.intensityRelThresh;
    rightRel = opts.intensityRelThresh;
    leftAbs = opts.intensityAbsThresh;
    rightAbs = opts.intensityAbsThresh;
    if ~isnan(opts.leftRelThresh),  leftRel  = opts.leftRelThresh;  end
    if ~isnan(opts.rightRelThresh), rightRel = opts.rightRelThresh; end
    if ~isnan(opts.leftAbsThresh),  leftAbs  = opts.leftAbsThresh;  end
    if ~isnan(opts.rightAbsThresh), rightAbs = opts.rightAbsThresh; end
    assert(leftRel >= 0 && rightRel >= 0, 'leftRelThresh/rightRelThresh must be nonnegative.');
    assert(leftAbs >= 0 && rightAbs >= 0, 'leftAbsThresh/rightAbsThresh must be nonnegative.');

    mag_left = sqrt(I_left);
    mag_right = sqrt(I_right);

    finiteZ = isfinite(real(Z_est)) & isfinite(imag(Z_est));
    finiteL = isfinite(mag_left);
    finiteR = isfinite(mag_right);

    magLeftMax = max(mag_left(finiteL), [], 'all');
    magRightMax = max(mag_right(finiteR), [], 'all');
    if isempty(magLeftMax) || magLeftMax <= 0
        error('I_left has no positive finite pixels.');
    end
    if isempty(magRightMax) || magRightMax <= 0
        error('I_right has no positive finite pixels.');
    end

    absZ = abs(Z_est);
    zMax = max(absZ(finiteZ), [], 'all');
    if isempty(zMax) || ~isfinite(zMax)
        zMax = 0;
    end

    leftFloor  = max(leftAbs,  leftRel  * magLeftMax);
    rightFloor = max(rightAbs, rightRel * magRightMax);
    zFloor     = max(opts.zAbsThresh, opts.zRelThresh * zMax);

    K_left = finiteL & (mag_left > leftFloor);
    K_right = finiteR & (mag_right > rightFloor);

    if nnz(K_left) == 0
        error('No valid focal pixels in K_left. Relax thresholds or check I_left.');
    end
    if nnz(K_right) == 0
        error('No valid focal pixels in K_right. Relax thresholds or check I_right.');
    end

    crossBase = K_left & K_right & finiteZ;
    if zFloor > 0
        crossBase = crossBase & (absZ > zFloor);
    end

    if isempty(opts.crossMask)
        crossMask = crossBase;
    else
        assert(isequal(size(opts.crossMask), [H,W]) && islogical(opts.crossMask), ...
            'crossMask must be an HxW logical array.');
        crossMask = opts.crossMask & crossBase;
    end

    idxCross = find(crossMask);
    nCross = numel(idxCross);
    if nCross == 0
        error('crossMask is empty after applying validity/threshold masks.');
    end

    % The anchor must lie in crossMask so that the fixed global phase also
    % couples the right-field scale through the Z equation.
    if isempty(opts.anchor)
        anchor_idx = choose_anchor(crossMask, mag_left, mag_right, Z_est, opts.anchorStrategy);
    else
        anchor_idx = opts.anchor;
        assert(isscalar(anchor_idx) && anchor_idx >= 1 && anchor_idx <= H*W, ...
            'opts.anchor must be a valid linear index.');
        assert(crossMask(anchor_idx), 'opts.anchor must lie inside crossMask.');
    end

    anchor_mask = false(H,W);
    anchor_mask(anchor_idx) = true;

    K_left_free = K_left;
    K_left_free(anchor_idx) = false;

    nLeft = nnz(K_left_free);
    nRight = nnz(K_right);
    nKLeft = nnz(K_left);
    nKRight = nnz(K_right);

    W_left = zeros(H,W);
    W_right = zeros(H,W);
    W_cross = zeros(H,W);
    W_left(K_left) = mag_left(K_left);
    W_right(K_right) = mag_right(K_right);
    W_cross(crossBase) = mag_left(crossBase) .* mag_right(crossBase);

    idxLeft_out = find(~supp_left);
    idxRight_out = find(~supp_right);
    nLeftOut = numel(idxLeft_out);
    nRightOut = numel(idxRight_out);

    % Fixed left-anchor phasor and its contribution to homogeneous blocks.
    Xfix = zeros(H,W);
    Xfix(anchor_idx) = opts.anchorVal;

    Eleft_fix = ifft2c(W_left .* Xfix);
    bLeft = -opts.supportScale * Eleft_fix(idxLeft_out);

    bRight = zeros(nRightOut, 1);

    Rfix = Z_est .* Xfix;
    bCross = -opts.crossWeight * Rfix(idxCross);

    m = nLeftOut + nRightOut + nCross + ...
        (lambdaLeft > 0)*nLeft + (lambdaRight > 0)*nRight;
    n = nLeft + nRight;

    toFullLeftFree = @(xleft) embed_on_mask(xleft, K_left_free, H, W);
    toVecLeftFree  = @(X) X(K_left_free);
    toFullRight    = @(xright) embed_on_mask(xright, K_right, H, W);
    toVecRight     = @(X) X(K_right);

    Amul = @(xvec) local_A_times(xvec, nLeft, nRight, H, W, ...
        idxLeft_out, idxRight_out, idxCross, ...
        W_left, W_right, W_cross, Z_est, ...
        opts.supportScale, opts.crossWeight, ...
        lambdaLeft, lambdaRight, toFullLeftFree, toFullRight);

    ATmul = @(yvec) local_AH_times(yvec, nLeft, nRight, H, W, ...
        idxLeft_out, idxRight_out, idxCross, ...
        W_left, W_right, W_cross, Z_est, ...
        opts.supportScale, opts.crossWeight, ...
        lambdaLeft, lambdaRight, toVecLeftFree, toVecRight);

    b = [bLeft; bRight; bCross];
    if lambdaLeft > 0
        b = [b; zeros(nLeft,1)];
    end
    if lambdaRight > 0
        b = [b; zeros(nRight,1)];
    end

    sys = struct();
    sys.Amul = Amul;
    sys.ATmul = ATmul;
    sys.b = b;
    sys.m = m;
    sys.n = n;
    sys.nLeft = nLeft;          % free left unknowns only
    sys.nRight = nRight;
    sys.nKLeft = nKLeft;        % includes hard anchor
    sys.nKRight = nKRight;
    sys.K_left = K_left;
    sys.K_right = K_right;
    sys.K_left_free = K_left_free;
    sys.crossMask = crossMask;
    sys.crossBase = crossBase;
    sys.idxCross = idxCross;
    sys.idxLeft_out = idxLeft_out;
    sys.idxRight_out = idxRight_out;
    sys.H = H;
    sys.W = W;
    sys.I_left = I_left;
    sys.I_right = I_right;
    sys.mag_left = mag_left;
    sys.mag_right = mag_right;
    sys.Z_est = Z_est;
    sys.C = Z_est;
    sys.W_left = W_left;
    sys.W_right = W_right;
    sys.W_cross = W_cross;
    sys.supp_left = supp_left;
    sys.supp_right = supp_right;
    sys.anchor_idx = anchor_idx;
    sys.anchor_mask = anchor_mask;
    sys.anchorVal = opts.anchorVal;
    sys.hardAnchor = true;
    sys.leftFloor = leftFloor;
    sys.rightFloor = rightFloor;
    sys.zFloor = zFloor;
    sys.lambdaLeft = lambdaLeft;
    sys.lambdaRight = lambdaRight;
    sys.supportScale = opts.supportScale;
    sys.crossWeight = opts.crossWeight;
    sys.toFullLeftFree = toFullLeftFree;
    sys.toVecLeftFree = toVecLeftFree;
    sys.toFullRight = toFullRight;
    sys.toVecRight = toVecRight;
    sys.fftConvention = 'centered-unitary';
    sys.description = ['Vanilla DBFH hard-anchor system. Unknowns are ', ...
        'X_left on K_left_free and X_right on K_right; cross term convention is ', ...
        'Z_est = conj(A_left).*A_right.'];
end

% =====================================================================
% Forward operator: linear part only. Fixed anchor is accounted for in b.
% =====================================================================
function y = local_A_times(xvec, nLeft, nRight, H, W, ...
    idxLeft_out, idxRight_out, idxCross, ...
    W_left, W_right, W_cross, Z_est, supportScale, crossWeight, ...
    lambdaLeft, lambdaRight, toFullLeftFree, toFullRight)

    assert(numel(xvec) == nLeft+nRight, 'xvec length must equal sys.n.');

    xleft = xvec(1:nLeft);
    xright = xvec(nLeft+1:nLeft+nRight);

    X_left_free = toFullLeftFree(xleft);
    X_right = toFullRight(xright);

    E_left = ifft2c(W_left .* X_left_free);
    yLeft = supportScale * E_left(idxLeft_out);

    E_right = ifft2c(W_right .* X_right);
    yRight = supportScale * E_right(idxRight_out);

    R = Z_est .* X_left_free - W_cross .* X_right;
    yCross = crossWeight * R(idxCross);

    y = [yLeft; yRight; yCross];

    if lambdaLeft > 0
        y = [y; lambdaLeft * xleft];
    end
    if lambdaRight > 0
        y = [y; lambdaRight * xright];
    end
end

% =====================================================================
% Adjoint operator of the linear part.
% =====================================================================
function x = local_AH_times(yvec, nLeft, nRight, H, W, ...
    idxLeft_out, idxRight_out, idxCross, ...
    W_left, W_right, W_cross, Z_est, supportScale, crossWeight, ...
    lambdaLeft, lambdaRight, toVecLeftFree, toVecRight)

    nLeftOut = numel(idxLeft_out);
    nRightOut = numel(idxRight_out);
    nCross = numel(idxCross);

    ptr = 0;
    yLeft = yvec(ptr+1:ptr+nLeftOut); ptr = ptr+nLeftOut;
    yRight = yvec(ptr+1:ptr+nRightOut); ptr = ptr+nRightOut;
    yCross = yvec(ptr+1:ptr+nCross); ptr = ptr+nCross;

    if lambdaLeft > 0
        yLamLeft = yvec(ptr+1:ptr+nLeft); ptr = ptr+nLeft;
    else
        yLamLeft = [];
    end

    if lambdaRight > 0
        yLamRight = yvec(ptr+1:ptr+nRight);
    else
        yLamRight = [];
    end

    Eleft_out = zeros(H,W, 'like', yvec);
    Eleft_out(idxLeft_out) = yLeft;
    gLeft_support = supportScale * conj(W_left) .* fft2c(Eleft_out);

    Eright_out = zeros(H,W, 'like', yvec);
    Eright_out(idxRight_out) = yRight;
    gRight_support = supportScale * conj(W_right) .* fft2c(Eright_out);

    Y = zeros(H,W, 'like', yvec);
    Y(idxCross) = yCross;

    gLeft_cross = crossWeight * conj(Z_est) .* Y;
    gRight_cross = -crossWeight * conj(W_cross) .* Y;

    gLeft_vec = toVecLeftFree(gLeft_support + gLeft_cross);
    gRight_vec = toVecRight(gRight_support + gRight_cross);

    if lambdaLeft > 0
        gLeft_vec = gLeft_vec + lambdaLeft * yLamLeft;
    end
    if lambdaRight > 0
        gRight_vec = gRight_vec + lambdaRight * yLamRight;
    end

    x = [gLeft_vec; gRight_vec];
end

% =====================================================================
% Helpers
% =====================================================================
function X = embed_on_mask(x, mask, H, W)
    X = zeros(H,W, 'like', x);
    X(mask) = x;
end

function anchor_idx = choose_anchor(mask, mag_left, mag_right, Z_est, anchorStrategy)
    idx = find(mask);
    strategy = lower(anchorStrategy);

    switch strategy
        case 'median'
            [H,W] = size(mask);
            [r,c] = ind2sub([H,W], idx);
            [~,ii] = min((r - median(r)).^2 + (c - median(c)).^2);
            anchor_idx = idx(ii);

        case 'maxmagleft'
            score = mag_left(idx);
            [~,ii] = max(score);
            anchor_idx = idx(ii);

        case 'maxmagright'
            score = mag_right(idx);
            [~,ii] = max(score);
            anchor_idx = idx(ii);

        case 'maxcombined'
            q = abs(Z_est);
            qMax = max(q(idx));
            if qMax > 0
                score = mag_left(idx) .* mag_right(idx) .* (q(idx) / qMax);
            else
                score = mag_left(idx) .* mag_right(idx);
            end
            [~,ii] = max(score);
            anchor_idx = idx(ii);

        otherwise
            error('Unknown anchorStrategy: %s. Use maxCombined, maxMagLeft, maxMagRight, or median.', anchorStrategy);
    end
end
