function out = diff_DBFH(I_full, C, cs_E, cs_dE, varargin)
%DIFF_DBFH Differential DBFH solve via prepared reduced operator system.
%
% Solves the reduced differential DBFH system using diff_dbfh_prepare_system:
%
%   C .* X - sqrt(I) .* deltaA = 0        on K
%   E(outside cs_E) = 0
%   real(X(anchor)) = anchorVal
%
% with
%   A      = sqrt(I) .* X
%   E      = ifft2c(A)
%   deltaE = m .* E,      m = double(cs_dE)
%   deltaA = fft2c(deltaE)
%
% Unknown is solved only on a reduced focal-plane mask K.
%
% Name-value options
% ------------------
% 'K'            : logical mask of unknown support in focal plane
% 'KRelThresh'   : relative threshold for automatic K         (default 1e-3)
% 'LambdaCS'     : weight on CS block                         (default 1)
% 'LambdaRef'    : weight on anchor row                       (default 1e-3)
% 'AnchorIndex'  : full-grid linear anchor index             (default brightest pixel in K)
% 'AnchorVal'    : RHS of anchor row real(X(anchor))         (default 1)
% 'LSQRTol'      : LSQR tolerance                            (default 1e-8)
% 'LSQRMaxIter'  : LSQR max iterations                       (default 200)
% 'X0'           : initial guess, either full-grid or K-only (default ones on K)
% 'NormalizeX'   : normalize to unit modulus after solve     (default true)
% 'Verbose'      : print diagnostics                         (default true)
% 'ShowPlots'    : show LSQR residual plot                   (default false)
% 'Debug_X_true' : optional truth for diagnostics            (default [])
%
% Output
% ------
% out.X          : recovered full-grid X (raw LSQR solution)
% out.X_norm     : unit-modulus version of X if NormalizeX=true, else = X
% out.A, out.E, out.deltaE, out.deltaA
% out.flag, out.relres, out.iter, out.resvec
% out.res_eq, out.res_cs, out.res_ref
% out.sys        : prepared system struct

p = inputParser;
p.addParameter('K', [], @(x) isempty(x) || islogical(x));
p.addParameter('KRelThresh', 1e-3, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('LambdaCS', 1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('LambdaRef', 1e-3, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('AnchorIndex', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
p.addParameter('AnchorVal', 1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('LSQRTol', 1e-8, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('LSQRMaxIter', 200, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('X0', [], @(x) isnumeric(x));
p.addParameter('NormalizeX', true, @(x) islogical(x) && isscalar(x));
p.addParameter('Verbose', true, @(x) islogical(x) && isscalar(x));
p.addParameter('ShowPlots', false, @(x) islogical(x) && isscalar(x));
p.addParameter('Debug_X_true', [], @(x) isnumeric(x) || isempty(x));
p.parse(varargin{:});

opts = p.Results;

I_full = double(I_full);
C      = double(C);
cs_E   = logical(cs_E);
cs_dE  = logical(cs_dE);

[Ny, Nx] = size(I_full);
if ~isequal(size(C), [Ny, Nx]) || ~isequal(size(cs_E), [Ny, Nx]) || ~isequal(size(cs_dE), [Ny, Nx])
    error('I_full, C, cs_E, and cs_dE must all have identical size.');
end

% ----------------------------
% prepare reduced system
% ----------------------------
prep_opts = struct();
prep_opts.KRelThresh = opts.KRelThresh;
prep_opts.lambdaCS   = opts.LambdaCS;
prep_opts.lambdaRef  = opts.LambdaRef;
prep_opts.anchorVal  = opts.AnchorVal;

if ~isempty(opts.K)
    prep_opts.K = opts.K;
end
if ~isempty(opts.AnchorIndex)
    prep_opts.anchor = opts.AnchorIndex;
end

sys = diff_dbfh_prepare_system(I_full, C, cs_E, cs_dE, prep_opts);

% ----------------------------
% initial guess on reduced support
% ----------------------------
if isempty(opts.X0)
    x0 = ones(sys.n, 1);
else
    X0 = opts.X0;
    if isequal(size(X0), [Ny, Nx])
        x0 = X0(sys.K);
    elseif isequal(size(X0), [sys.n, 1]) || isequal(size(X0), [1, sys.n])
        x0 = X0(:);
    else
        error('X0 must be either Ny x Nx or nnz(K) x 1.');
    end
end

% ----------------------------
% solve
% ----------------------------
[x, flag, relres, iter, resvec] = lsqr(sys.lsqrA, sys.b, opts.LSQRTol, opts.LSQRMaxIter, [], [], x0);

% ----------------------------
% reconstruct full-grid X
% ----------------------------
X = zeros(Ny, Nx);
X(sys.K) = x;

if opts.NormalizeX
    X_norm = X ./ max(abs(X), eps);
else
    X_norm = X;
end

sqrtI = sys.sqrtI;
m = sys.mMask;
mask_out_E = ~cs_E;

A = sqrtI .* X_norm;
E = ifft2c(A);
deltaE = m .* E;
deltaA = fft2c(deltaE);

% residuals
r_eq_full = C .* X_norm - sqrtI .* deltaA;
r_eq = r_eq_full(sys.K);
r_cs = E(mask_out_E);
r_ref = real(X_norm(sys.anchor_idx)) - opts.AnchorVal;

% ----------------------------
% package output
% ----------------------------
out = struct();
out.X = X;
out.X_norm = X_norm;
out.A = A;
out.E = E;
out.deltaE = deltaE;
out.deltaA = deltaA;

out.flag = flag;
out.relres = relres;
out.iter = iter;
out.resvec = resvec;

out.K = sys.K;
out.anchor_idx = sys.anchor_idx;
out.res_eq = norm(r_eq(:));
out.res_cs = norm(r_cs(:));
out.res_ref = abs(r_ref);

out.sys = sys;

% optional debug against truth
if ~isempty(opts.Debug_X_true)
    X_true = opts.Debug_X_true;
    if ~isequal(size(X_true), [Ny, Nx])
        error('Debug_X_true must be Ny x Nx.');
    end

    A_true = sqrtI .* X_true;
    E_true = ifft2c(A_true);
    deltaE_true = m .* E_true;
    deltaA_true = fft2c(deltaE_true);

    r_eq_true_full = C .* X_true - sqrtI .* deltaA_true;
    r_eq_true = r_eq_true_full(sys.K);
    r_cs_true = E_true(mask_out_E);
    r_ref_true = real(X_true(sys.anchor_idx)) - opts.AnchorVal;

    out.debug = struct();
    out.debug.r_eq_true = norm(r_eq_true(:));
    out.debug.r_cs_true = norm(r_cs_true(:));
    out.debug.r_ref_true = abs(r_ref_true);
end

% ----------------------------
% diagnostics
% ----------------------------
if opts.Verbose
    fprintf('diff_DBFH: %d x %d\n', Ny, Nx);
    fprintf('  nnz(K)       = %d / %d (%.2f%%)\n', nnz(sys.K), numel(sys.K), 100*nnz(sys.K)/numel(sys.K));
    fprintf('  anchor idx   = %d\n', sys.anchor_idx);
    fprintf('  lsqr flag    = %d\n', flag);
    fprintf('  lsqr relres  = %.3e\n', relres);
    fprintf('  lsqr iter    = %d\n', iter);
    fprintf('  ||eq||       = %.3e\n', out.res_eq);
    fprintf('  ||cs||       = %.3e\n', out.res_cs);
    fprintf('  |ref|        = %.3e\n', out.res_ref);

    if isfield(out, 'debug')
        fprintf('  ||eq_true||  = %.3e\n', out.debug.r_eq_true);
        fprintf('  ||cs_true||  = %.3e\n', out.debug.r_cs_true);
        fprintf('  |ref_true|   = %.3e\n', out.debug.r_ref_true);
    end
end

if opts.ShowPlots && ~isempty(resvec)
    figure('Color','w');
    semilogy(resvec, 'LineWidth', 2);
    grid on;
    xlabel('LSQR iteration');
    ylabel('Residual norm');
    title('diff\_DBFH residual vs iteration');
end

end

% 
% function F = fft2c(x)
% F = fftshift(fft2(ifftshift(x))) / sqrt(numel(x));
% end
% 
% function x = ifft2c(F)
% x = fftshift(ifft2(ifftshift(F))) * sqrt(numel(F));
% end
