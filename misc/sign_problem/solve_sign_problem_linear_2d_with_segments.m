function out = solve_sign_problem_linear_2d_with_segments(absz, rez, support_mask_x, seg_id, varargin)
% Pixelwise sign solve with added equality constraints inside segments.
%
% Unknown is still X per pixel.
%
% Minimize:
%   || A_support X - b ||^2 + lambda_seg^2 * || A_seg X ||^2
%
% where A_seg encodes X(i)-X(j)=0 for neighboring pixels in same segment.

p = inputParser;
p.addParameter('Tol', 1e-6, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('MaxIter', 200, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('Verbose', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('UseSingle', false, @(x) islogical(x) || isnumeric(x));
p.addParameter('LambdaSeg', 1.0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('ShowPlots', false, @(x) islogical(x)|| isnumeric(x));
p.parse(varargin{:});

tol = p.Results.Tol;
maxIter = p.Results.MaxIter;
verbose = logical(p.Results.Verbose);
useSingle = logical(p.Results.UseSingle);
lambdaSeg = p.Results.LambdaSeg;
show_plots = p.Results.ShowPlots;

if ~isequal(size(absz), size(rez), size(support_mask_x), size(seg_id))
    error('absz, rez, support_mask_x, seg_id must have the same size.');
end

if useSingle
    absz = single(absz);
    rez  = single(rez);
else
    absz = double(absz);
    rez  = double(rez);
end
support_mask_x = logical(support_mask_x);

[Ny, Nx] = size(absz);
Npix = Ny * Nx;
outside_mask = ~support_mask_x;
nOut = nnz(outside_mask);

% Centered FFT convention
F2  = @(x) fftshift(fft2(ifftshift(x)));
iF2 = @(X) fftshift(ifft2(ifftshift(X)));

% q = sqrt(|z|^2 - Re(z)^2)
q2 = absz.^2 - rez.^2;
q2(q2<0) = 0; % in noisy scenarios can get negative values
tolq = cast(1e-12 * max(1, double(max(absz(:)).^2)), class(absz));
q2(q2 < 0 & q2 > -tolq) = 0;
if any(q2(:) < 0)
    error('|z|^2 - Re(z)^2 is negative beyond numerical tolerance.');
end
q = sqrt(q2);

% Build segment edge constraints
[edge_i, edge_j] = build_segment_edges(seg_id);
nEdges = numel(edge_i);

if verbose
    fprintf('Image size:     %d x %d\n', Ny, Nx);
    fprintf('Unknowns:       %d\n', Npix);
    fprintf('Outside pts:    %d\n', nOut);
    fprintf('Segment edges:  %d\n', nEdges);
end

% RHS
rhs_full = 1i * iF2(rez);
rhs_out  = rhs_full(outside_mask);

if useSingle
    b = single([real(rhs_out); imag(rhs_out); zeros(nEdges,1,'single')]);
else
    b = [real(rhs_out); imag(rhs_out); zeros(nEdges,1)];
end

Afun = @linop;
x0 = zeros(Npix,1, class(q));

[x_relaxed_vec, flag, relres, iter, resvec] = lsqr(Afun, b, tol, maxIter, [], [], x0);

X_relaxed = reshape(x_relaxed_vec, Ny, Nx);
X_sign = sign(X_relaxed);
X_sign(X_sign == 0) = 1;

z_relaxed = rez + 1i * (X_relaxed .* q);
z_sign    = rez + 1i * (X_sign    .* q);

g_relaxed = iF2(z_relaxed);
g_sign    = iF2(z_sign);

out = struct();
out.q = q;
out.X_relaxed = X_relaxed;
out.X_sign = X_sign;
out.z_relaxed = z_relaxed;
out.z_sign = z_sign;
out.g_relaxed = g_relaxed;
out.g_sign = g_sign;
out.flag = flag;
out.relres = relres;
out.iter = iter;
out.resvec = resvec;
out.edge_i = edge_i;
out.edge_j = edge_j;
out.lambda_seg = lambdaSeg;

if show_plots
    figure;semilogy(resvec,'linewidth',2);title({'sign problem';'residual vs # iteration'});grid on;xlim([1,maxIter]);
end


    function y = linop(x, transp_flag)
        switch lower(transp_flag)
            case 'notransp'
                X = reshape(x, Ny, Nx);

                % Original support equations
                Y = iF2(q .* X);
                Yout = Y(outside_mask);
                y1 = real(Yout);
                y2 = imag(Yout);

                % Segment equality equations
                xv = X(:);
                y3 = lambdaSeg * (xv(edge_i) - xv(edge_j));

                y = [y1; y2; y3];

            case 'transp'
                % split residual vector
                u = x(:);
                u1 = u(1:nOut);
                u2 = u(nOut+1:2*nOut);
                u3 = u(2*nOut+1:end);

                % Adjoint of original support equations
                U = zeros(Ny, Nx, 'like', q);
                U(outside_mask) = u1 + 1i*u2;
                T = F2(U) / (Ny * Nx);
                g1 = real(q .* T);
                g1 = g1(:);

                % Adjoint of segment equality equations
                g2 = zeros(Npix,1, 'like', g1);
                if nEdges > 0
                    ru = lambdaSeg * u3;
                    zfill = cast(0, 'like', ru);
                    g2 = g2 + cast(accumarray(edge_i,  ru, [Npix 1], @sum, zfill), 'like', g2);
                    g2 = g2 + cast(accumarray(edge_j, -ru, [Npix 1], @sum, zfill), 'like', g2);
                end
                y = g1 + g2;

            otherwise
                error('transp_flag must be ''notransp'' or ''transp''.');
        end
    end
end