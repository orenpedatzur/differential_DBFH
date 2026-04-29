function out = diff_dbfh_unpack_solution(sys, x1vec, opts)
%DIFF_DBFH_UNPACK_SOLUTION  Convert solved X1 vector into fields and diagnostics.
%
%   out = diff_dbfh_unpack_solution(sys, x1vec)
%
% x1vec is the complex solution on sys.K, after undoing any right
% preconditioner. If the LSQR variable is u, first compute x = Minv(u),
% then convert the real-stacked x into complex x1vec.
%
% IMPORTANT FOURIER CONVENTION
%   This version assumes CENTERED arrays and reconstructs pupil fields using:
%       ifft2c(F) = fftshift(ifft2(ifftshift(F)))
%
%   Using raw ifft2(F) here will introduce a checkerboard/linear phase factor
%   when F is stored on a centered Fourier grid.

    arguments
        sys struct
        x1vec
        opts.normalizePhasor (1,1) logical = true
    end

    assert(numel(x1vec) == sys.n, 'x1vec must have sys.n elements.');

    H = sys.H;
    W = sys.W;

    X1_raw = zeros(H,W);
    X1_raw(sys.K) = x1vec(:);

    if opts.normalizePhasor
        amp = abs(X1_raw);
        X1 = zeros(H,W);
        valid = sys.K & (amp > 0);
        X1(valid) = X1_raw(valid) ./ amp(valid);
    else
        X1 = X1_raw;
    end

    Cabs = abs(sys.C);
    validC = sys.K & isfinite(Cabs) & (Cabs > max(sys.cFloor, eps));
    Cphase = zeros(H,W);
    Cphase(validC) = sys.C(validC) ./ Cabs(validC);

    % From C = A1^* A2 = |A1||A2| conj(X1) X2:
    % C/|C| = conj(X1) X2, hence X2 = (C/|C|) X1.
    X2 = zeros(H,W);
    X2(validC) = Cphase(validC) .* X1(validC);

    A1 = sys.W1 .* X1;
    A2 = sys.W2eff .* X1;

    % CENTERED inverse FFT. Raw ifft2(A1) is wrong for centered focal arrays.
    E1 = ifft2c_local(A1);
    E2 = ifft2c_local(A2);
    E  = E1 + E2;

    nrm1 = max(norm(E1(:)), eps);
    nrm2 = max(norm(E2(:)), eps);

    out = struct();
    out.X1 = X1;
    out.X1_raw = X1_raw;
    out.X2 = X2;
    out.A1 = A1;
    out.A2 = A2;
    out.E1 = E1;
    out.E2 = E2;
    out.E = E;
    out.validC = validC;
    out.leak1 = norm(E1(~sys.supp1)) / nrm1;
    out.leak2 = norm(E2(~sys.supp2)) / nrm2;
    out.raw_phasor_abs_median = median(abs(X1_raw(sys.K)));
    out.raw_phasor_abs_mad = median(abs(abs(X1_raw(sys.K)) - out.raw_phasor_abs_median));
end

function E = ifft2c_local(F)
    E = fftshift(ifft2(ifftshift(F)));
end
