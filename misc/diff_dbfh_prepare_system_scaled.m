function sys = diff_dbfh_prepare_system_scaled(mag1, C, supp1, supp2, opts)
%DIFF_DBFH_PREPARE_SYSTEM_SCALED  Differential-DBFH system with row-scaled support equations.
%
%   sys = diff_dbfh_prepare_system(mag1, C, supp1, supp2, opts)
%
% IMPORTANT FOURIER CONVENTION
%   This version assumes CENTERED arrays:
%       - mag1, C, and X1 live on a centered Fourier/focal grid
%       - supp1 and supp2 live on a centered pupil grid
%
%   Therefore all support equations use:
%       ifft2c(F) = fftshift(ifft2(ifftshift(F)))
%       fft2c(E)  = fftshift(fft2(ifftshift(E)))
%
%   The adjoint uses fft2c(.)/(H*W), because MATLAB ifft2 includes 1/(H*W).
%
% Unknown:
%   x(k) = X1(k) = exp(1i*phi1(k)), the phasor of A1, on frequency mask K.
%
% Differential-DBFH equations from C = A1^* A2:
%   ifft2c( |A1| .* X1 )        = 0 outside supp1
%   ifft2c( (C ./ |A1|) .* X1 ) = 0 outside supp2
%
% because:
%   (C ./ |A1|) .* X1 = |A2| X2.
%
% Inputs
%   mag1  : HxW double, nonnegative, centered |A1| = sqrt(I_blocked)
%   C     : HxW complex double, centered recovered cross term A1^* A2
%   supp1 : HxW logical, centered pupil support of E1, true = inside support
%   supp2 : HxW logical, centered pupil support of E2/blocker region, true = inside support
%
% Name-value options
%   lambda               default 0       Tikhonov weight, adds lambda*I rows
%   anchor               default []      linear index in K to anchor
%   anchorVal            default 1+0i    value imposed at anchor
%   mag1RelThresh        default 1e-12   K threshold relative to max(mag1)
%   mag1AbsThresh        default 0       absolute K threshold on mag1
%   cRelThresh           default 0       optional K threshold relative to max(abs(C))
%   cAbsThresh           default 0       optional absolute K threshold on abs(C)
%   anchorStrategy       default 'maxMag1'; options: 'maxMag1', 'median', 'maxCombined'
%   warnOnSupportOverlap default true    warn if supp1 and supp2 overlap
%
% Output sys has the same operator interface as dbh_prepare_system:
%   sys.Amul(z)    -> y = A*z        complex operator
%   sys.ATmul(y)   -> x = A'*y       complex adjoint
%   sys.b          -> RHS, complex
%   sys.m, sys.n   -> sizes
%   sys.K          -> unknown frequency mask
%
% Compatible with:
%   hutchinson_diag_AtA
%   right_diag_precond_from_diag
%   make_Afun_lsqr_stacked

    arguments
        mag1 double
        C double
        supp1 logical
        supp2 logical
        opts.lambda (1,1) double {mustBeNonnegative} = 0
        opts.anchor = []
        opts.anchorVal (1,1) double = 1+0i
        opts.mag1RelThresh (1,1) double {mustBeNonnegative} = 1e-12
        opts.mag1AbsThresh (1,1) double {mustBeNonnegative} = 0
        opts.cRelThresh (1,1) double {mustBeNonnegative} = 0
        opts.cAbsThresh (1,1) double {mustBeNonnegative} = 0
        opts.anchorStrategy char = 'maxMag1'
        opts.warnOnSupportOverlap (1,1) logical = true
        opts.supportScale (1,1) double = NaN
        opts.anchorWeight (1,1) double {mustBePositive} = 1
    end

    % ---- sanity / shared sizes ----
    [H,W] = size(mag1);
    if isnan(opts.supportScale)
        supportScale = sqrt(H*W);   % unitary-FFT-equivalent scaling for support rows
    else
        assert(opts.supportScale > 0, 'supportScale must be positive.');
        supportScale = opts.supportScale;
    end
    anchorWeight = opts.anchorWeight;
    assert(isequal(size(mag1), size(C), size(supp1), size(supp2)), ...
        'mag1, C, supp1, and supp2 must be HxW and of equal size.');
    assert(isreal(mag1), 'mag1 must be real and nonnegative.');
    assert(all(mag1(:) >= 0 | isnan(mag1(:))), 'mag1 must be nonnegative.');

    if opts.warnOnSupportOverlap && any(supp1(:) & supp2(:))
        warning('diff_dbfh_prepare_system:SupportOverlap', ...
            'supp1 and supp2 overlap. The memo assumes disjoint pupil supports; verify this is intentional.');
    end

    finiteC = isfinite(real(C)) & isfinite(imag(C));
    mag1Max = max(mag1(isfinite(mag1)), [], 'all');
    if isempty(mag1Max) || mag1Max <= 0
        error('mag1 has no positive finite pixels.');
    end

    absC = abs(C);
    cMax = max(absC(finiteC), [], 'all');
    if isempty(cMax) || ~isfinite(cMax)
        cMax = 0;
    end

    mag1Floor = max(opts.mag1AbsThresh, opts.mag1RelThresh * mag1Max);
    cFloor    = max(opts.cAbsThresh,    opts.cRelThresh    * cMax);

    % Unknown mask. Do not divide by |A1| below mag1Floor.
    K = isfinite(mag1) & finiteC & (mag1 > mag1Floor);
    if cFloor > 0
        K = K & (absC > cFloor);
    end

    nK = nnz(K);
    if nK == 0
        error('No valid frequency pixels in K. Relax thresholds or check mag1/C.');
    end

    % Spectral weights for the two support blocks.
    % Block 1 reconstructs A1 = |A1| X1.
    % Block 2 reconstructs A2 = (C/|A1|) X1.
    W1 = zeros(H,W);
    W2eff = zeros(H,W);
    W1(K) = mag1(K);
    W2eff(K) = C(K) ./ mag1(K);

    idx1_out = find(~supp1);
    idx2_out = find(~supp2);
    n1 = numel(idx1_out);
    n2 = numel(idx2_out);

    % Anchor: fixes the remaining global phase of X1.
    if isempty(opts.anchor)
        anchor_idx = choose_anchor(K, mag1, W2eff, opts.anchorStrategy);
    else
        anchor_idx = opts.anchor;
        assert(isscalar(anchor_idx) && anchor_idx >= 1 && anchor_idx <= H*W, ...
            'opts.anchor must be a valid linear index.');
        assert(K(anchor_idx), 'opts.anchor must lie inside K.');
    end
    anchor_mask = false(H,W);
    anchor_mask(anchor_idx) = true;

    m = n1 + n2 + 1 + (opts.lambda > 0) * nK;
    n = nK;

    toFull = @(zvec) embed_on_mask(zvec, K, H, W);
    toVec  = @(X) X(K);

    scale = H * W;  % adjoint scaling: ifft2c(.)^H = fft2c(.)/(H*W)

    Amul = @(zvec) local_A_times(zvec, idx1_out, idx2_out, ...
                                 W1, W2eff, anchor_mask, opts.lambda, toFull, ...
                                 supportScale, anchorWeight);

    ATmul = @(yvec) local_AH_times(yvec, H, W, idx1_out, idx2_out, ...
                                   W1, W2eff, anchor_mask, opts.lambda, ...
                                   toVec, scale, supportScale, anchorWeight);

    b = zeros(m,1);
    b(n1+n2+1) = anchorWeight * opts.anchorVal;

    % Store enough metadata for diagnostics and reconstruction.
    sys = struct();
    sys.Amul = Amul;
    sys.ATmul = ATmul;
    sys.b = b;
    sys.m = m;
    sys.n = n;
    sys.K = K;
    sys.idx1_out = idx1_out;
    sys.idx2_out = idx2_out;
    sys.idxA_out = idx1_out;   % compatibility aliases
    sys.idxB_out = idx2_out;
    sys.H = H;
    sys.W = W;
    sys.mag1 = mag1;
    sys.C = C;
    sys.W1 = W1;
    sys.W2eff = W2eff;
    sys.supp1 = supp1;
    sys.supp2 = supp2;
    sys.anchor_idx = anchor_idx;
    sys.anchorVal = opts.anchorVal;
    sys.mag1Floor = mag1Floor;
    sys.cFloor = cFloor;
    sys.lambda = opts.lambda;
    sys.supportScale = supportScale;
    sys.anchorWeight = anchorWeight;
    sys.toFull = toFull;
    sys.toVec = toVec;
    sys.fftConvention = 'centered';
    sys.description = 'Differential DBFH: centered convention; unknown is X1; block2 uses (C/|A1|)*X1.';
end

% ===== internal helpers =====

function y = local_A_times(zvec, idx1_out, idx2_out, W1, W2eff, anchor_mask, lambda, toFull, supportScale, anchorWeight)
    X1 = toFull(zvec);

    e1 = ifft2c_local(W1 .* X1);
    y1 = supportScale * e1(idx1_out);

    e2 = ifft2c_local(W2eff .* X1);
    y2 = supportScale * e2(idx2_out);

    yAnchor = anchorWeight * sum(X1(anchor_mask), 'all');

    y = [y1; y2; yAnchor];
    if lambda > 0
        y = [y; lambda * zvec];
    end
end

function x = local_AH_times(yvec, H, W, idx1_out, idx2_out, W1, W2eff, anchor_mask, lambda, toVec, scale, supportScale, anchorWeight)
    n1 = numel(idx1_out);
    n2 = numel(idx2_out);

    if lambda > 0
        y1 = yvec(1:n1);
        y2 = yvec(n1+1:n1+n2);
        yAnchor = yvec(n1+n2+1);
        yLam = yvec(n1+n2+2:end);
    else
        y1 = yvec(1:n1);
        y2 = yvec(n1+1:n1+n2);
        yAnchor = yvec(n1+n2+1);
    end

    E1out = zeros(H,W);
    E2out = zeros(H,W);
    E1out(idx1_out) = y1;
    E2out(idx2_out) = y2;

    % If forward rows are supportScale*ifft2c(.), adjoint gets the same supportScale.
    g1 = supportScale * conj(W1)    .* (fft2c_local(E1out) / scale);
    g2 = supportScale * conj(W2eff) .* (fft2c_local(E2out) / scale);
    g3 = zeros(H,W);
    g3(anchor_mask) = anchorWeight * yAnchor;

    G = g1 + g2 + g3;
    x = toVec(G);

    if lambda > 0
        x = x + lambda * yLam;
    end
end

function X = embed_on_mask(x, K, H, W)
    X = zeros(H,W, 'like', x);
    X(K) = x;
end

function anchor_idx = choose_anchor(K, mag1, W2eff, anchorStrategy)
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
            q = abs(W2eff);
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

function E = ifft2c_local(F)
    E = fftshift(ifft2(ifftshift(F)));
end

function F = fft2c_local(E)
    F = fftshift(fft2(ifftshift(E)));
end
