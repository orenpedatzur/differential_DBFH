function out = dbfh_unpack_solution(sys, xvec, opts)
%VANILLA_DBFH_UNPACK_SOLUTION
% Unpack a physical complex solution vector from vanilla_dbfh_prepare_system.
%
% xvec must be the physical solution after undoing any right preconditioner:
%       xvec = [x_left_free; x_right]
%
% The hard global-phase anchor is inserted exactly:
%       X_left(anchor) = sys.anchorVal
%
% Model convention used by vanilla_dbfh_prepare_system:
%       Z_est = conj(A_left) .* A_right
%
% Linear cross-term equation:
%       Z_est .* X_left - (|A_left| .* |A_right|) .* X_right = 0
%
% where:
%       A_left  = |A_left|  .* X_left
%       A_right = |A_right| .* X_right
%
% Assumes centered unitary FFT helpers fft2c/ifft2c are on the MATLAB path.

    arguments
        sys struct
        xvec
        opts.normalizePhasors (1,1) logical = false
        opts.normalizeLeft (1,1) logical = false
        opts.normalizeRight (1,1) logical = false
    end

    required = {'hardAnchor','n','nLeft','nRight','H','W', ...
        'K_left','K_left_free','K_right','anchor_idx','anchorVal', ...
        'mag_left','mag_right','W_cross','Z_est', ...
        'supp_left','supp_right','crossMask'};
    for k = 1:numel(required)
        assert(isfield(sys, required{k}), ...
            'sys is missing field "%s". Use a sys from vanilla_dbfh_prepare_system.', required{k});
    end

    assert(sys.hardAnchor, ...
        'sys must come from vanilla_dbfh_prepare_system with hardAnchor=true.');
    assert(numel(xvec) == sys.n, 'xvec must have sys.n elements.');

    nLeft = sys.nLeft;
    nRight = sys.nRight;
    H = sys.H;
    W = sys.W;

    xleft_vec = xvec(1:nLeft);
    xright_vec = xvec(nLeft+1:nLeft+nRight);

    % Raw focal-plane phasor estimates.
    X_left_raw = zeros(H,W, 'like', xvec);
    X_left_raw(sys.K_left_free) = xleft_vec(:);
    X_left_raw(sys.anchor_idx) = cast(sys.anchorVal, 'like', xvec);

    X_right_raw = zeros(H,W, 'like', xvec);
    X_right_raw(sys.K_right) = xright_vec(:);

    doNormLeft = opts.normalizePhasors || opts.normalizeLeft;
    doNormRight = opts.normalizePhasors || opts.normalizeRight;

    X_left = X_left_raw;
    if doNormLeft
        X_left = normalize_phasor_on_mask(X_left_raw, sys.K_left);
        % Preserve the selected global phase reference as a unit phasor.
        if abs(sys.anchorVal) > 0
            X_left(sys.anchor_idx) = cast(sys.anchorVal ./ abs(sys.anchorVal), 'like', xvec);
        end
    end

    X_right = X_right_raw;
    if doNormRight
        X_right = normalize_phasor_on_mask(X_right_raw, sys.K_right);
    end

    % Reconstructed focal-plane complex amplitudes.
    A_left = sys.mag_left .* X_left;
    A_right = sys.mag_right .* X_right;

    A_left_raw = sys.mag_left .* X_left_raw;
    A_right_raw = sys.mag_right .* X_right_raw;

    % Reconstructed pupil-plane fields.
    E_left = ifft2c(A_left);
    E_right = ifft2c(A_right);
    E = E_left + E_right;

    E_left_raw = ifft2c(A_left_raw);
    E_right_raw = ifft2c(A_right_raw);
    E_raw = E_left_raw + E_right_raw;

    % Linear cross residual: this is the residual of the system actually solved.
    cross_res_raw = sys.Z_est .* X_left_raw - sys.W_cross .* X_right_raw;
    cross_res = sys.Z_est .* X_left - sys.W_cross .* X_right;

    % Nonlinear direct cross-term diagnostic. This is useful after phasor
    % normalization, but it is not the linear objective minimized by LSQR.
    Z_from_fields = conj(A_left) .* A_right;
    Z_direct_res = sys.Z_est - Z_from_fields;

    Z_from_fields_raw = conj(A_left_raw) .* A_right_raw;
    Z_direct_res_raw = sys.Z_est - Z_from_fields_raw;

    cm = sys.crossMask;

    out = struct();

    % Phasors.
    out.X_left = X_left;
    out.X_right = X_right;
    out.X_left_raw = X_left_raw;
    out.X_right_raw = X_right_raw;

    % Focal fields.
    out.A_left = A_left;
    out.A_right = A_right;
    out.A_left_raw = A_left_raw;
    out.A_right_raw = A_right_raw;

    % Pupil fields.
    out.E_left = E_left;
    out.E_right = E_right;
    out.E = E;
    out.E_left_raw = E_left_raw;
    out.E_right_raw = E_right_raw;
    out.E_raw = E_raw;

    % Cross-term diagnostics.
    out.cross_residual = cross_res;
    out.cross_residual_raw = cross_res_raw;
    out.cross_relres = masked_relres(cross_res, cm, sys.Z_est .* X_left, sys.W_cross .* X_right);
    out.cross_relres_raw = masked_relres(cross_res_raw, cm, sys.Z_est .* X_left_raw, sys.W_cross .* X_right_raw);

    out.Z_from_fields = Z_from_fields;
    out.Z_from_fields_raw = Z_from_fields_raw;
    out.Z_direct_residual = Z_direct_res;
    out.Z_direct_residual_raw = Z_direct_res_raw;
    out.Z_direct_relres = norm(Z_direct_res(cm)) / max(norm(sys.Z_est(cm)), eps_like(xvec));
    out.Z_direct_relres_raw = norm(Z_direct_res_raw(cm)) / max(norm(sys.Z_est(cm)), eps_like(xvec));

    % Support leakage diagnostics.
    out.leak_left = norm(E_left(~sys.supp_left)) / max(norm(E_left(:)), eps_like(xvec));
    out.leak_right = norm(E_right(~sys.supp_right)) / max(norm(E_right(:)), eps_like(xvec));
    out.leak_total_outside_union = norm(E(~(sys.supp_left | sys.supp_right))) / max(norm(E(:)), eps_like(xvec));

    out.leak_left_raw = norm(E_left_raw(~sys.supp_left)) / max(norm(E_left_raw(:)), eps_like(xvec));
    out.leak_right_raw = norm(E_right_raw(~sys.supp_right)) / max(norm(E_right_raw(:)), eps_like(xvec));
    out.leak_total_outside_union_raw = norm(E_raw(~(sys.supp_left | sys.supp_right))) / max(norm(E_raw(:)), eps_like(xvec));

    % Raw phasor amplitude diagnostics. If these drift far from 1, the LSQR
    % solution is using amplitude freedom that the physical phasor model does
    % not really have.
    out.raw_left_phasor_abs_median = median(abs(X_left_raw(sys.K_left)));
    out.raw_left_phasor_abs_mad = median(abs(abs(X_left_raw(sys.K_left)) - out.raw_left_phasor_abs_median));
    out.raw_right_phasor_abs_median = median(abs(X_right_raw(sys.K_right)));
    out.raw_right_phasor_abs_mad = median(abs(abs(X_right_raw(sys.K_right)) - out.raw_right_phasor_abs_median));

    out.anchor_idx = sys.anchor_idx;
    out.anchorVal = sys.anchorVal;
    out.anchor_error_raw = abs(X_left_raw(sys.anchor_idx) - sys.anchorVal);
    out.normalized_left = doNormLeft;
    out.normalized_right = doNormRight;
end

% =====================================================================
% Helpers
% =====================================================================
function X = normalize_phasor_on_mask(Xraw, mask)
    X = zeros(size(Xraw), 'like', Xraw);
    amp = abs(Xraw);
    valid = mask & (amp > 0);
    X(valid) = Xraw(valid) ./ amp(valid);
end

function r = masked_relres(res, mask, lhs, rhs)
    denom = max(norm(lhs(mask)), norm(rhs(mask)));
    r = norm(res(mask)) / max(denom, eps_like(res));
end

function e = eps_like(x)
    if isa(x, 'gpuArray')
        if isaUnderlying(x, 'single')
            e = gpuArray(single(eps('single')));
        else
            e = gpuArray(eps);
        end
    elseif isa(x, 'single')
        e = single(eps('single'));
    else
        e = eps;
    end
end
