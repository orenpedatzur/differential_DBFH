function out = solve_sign_problem_linear(absz, rez, support_mask_x)
%SOLVE_SIGN_PROBLEM_LINEAR
% Linear relaxation of the sign problem:
%
%   Im(z) = X .* sqrt(|z|^2 - Re(z)^2)
%
% with support constraint on g(x) = ifft(z):
%
%   ifft(rez + 1i * X .* q)(x) = 0    for x outside support
%
% which implies
%
%   ifft(X .* q)(x) = 1i * ifft(rez)(x)   for x outside support
%
% This is solved as a linear least-squares problem for X.
% Then X is thresholded to sign(X) to produce a binary solution.
%
% INPUTS
%   absz            - Nx1 or 1xN real vector, |z|
%   rez             - same size real vector, Re(z)
%   support_mask_x  - same size logical vector, true inside compact support
%
% OUTPUT
%   out is a struct with:
%       .X_relaxed   : least-squares real-valued solution
%       .X_sign      : thresholded sign solution (+1/-1)
%       .q           : sqrt(|z|^2 - Re(z)^2)
%       .z_relaxed   : reconstructed z using X_relaxed
%       .z_sign      : reconstructed z using X_sign
%       .g_relaxed   : ifft(z_relaxed)
%       .g_sign      : ifft(z_sign)
%       .residual_ls : least-squares residual norm
%       .residual_pm : residual norm after sign thresholding
%       .A           : real linear system matrix
%       .b           : real linear system RHS
%
% NOTE
%   This is the linear version of your derivation. The true sign problem
%   requires X(k) in {-1,+1}; here we first solve the relaxed linear system
%   over R^N and then threshold.

    % --- force column vectors ---
    absz = absz(:);
    rez  = rez(:);
    support_mask_x = logical(support_mask_x(:));

    N = numel(absz);

    if numel(rez) ~= N || numel(support_mask_x) ~= N
        error('absz, rez, and support_mask_x must have the same length.');
    end

    % --- compute q = sqrt(|z|^2 - Re(z)^2) ---
    q2 = absz.^2 - rez.^2;
    small_neg = (q2 < 0) & (q2 > -1e-12 * max(1, max(absz).^2));
    q2(small_neg) = 0;

    if any(q2 < 0)
        error('|z|^2 - Re(z)^2 is negative beyond numerical tolerance.');
    end

    q = sqrt(q2);

    % --- build inverse DFT matrix consistent with ifft ---
    % ifft(z) = (1/N) * conj(W) * z
    F = conj(dftmtx(N)) / N;

    % rows outside support
    outside_idx = find(~support_mask_x);
    Fout = F(outside_idx, :);

    % complex linear system:
    %   Fout * diag(q) * X = 1i * Fout * rez
    Aq = Fout * diag(q);
    rhs = 1i * (Fout * rez);

    % X is real, so split into real system:
    % [Re(A); Im(A)] X = [Re(b); Im(b)]
    A = [real(Aq); imag(Aq)];
    b = [real(rhs); imag(rhs)];

    % solve least squares
    X_relaxed = A \ b;

    % threshold to +/-1
    X_sign = sign(X_relaxed);
    X_sign(X_sign == 0) = 1;

    % reconstructions
    imz_relaxed = X_relaxed .* q;
    imz_sign    = X_sign    .* q;

    z_relaxed = rez + 1i * imz_relaxed;
    z_sign    = rez + 1i * imz_sign;

    g_relaxed = ifft(z_relaxed);
    g_sign    = ifft(z_sign);

    % residuals in the real linear system
    residual_ls = norm(A * X_relaxed - b);
    residual_pm = norm(A * X_sign    - b);

    % output
    out = struct();
    out.X_relaxed   = X_relaxed;
    out.X_sign      = X_sign;
    out.q           = q;
    out.z_relaxed   = z_relaxed;
    out.z_sign      = z_sign;
    out.g_relaxed   = g_relaxed;
    out.g_sign      = g_sign;
    out.residual_ls = residual_ls;
    out.residual_pm = residual_pm;
    out.A           = A;
    out.b           = b;
end