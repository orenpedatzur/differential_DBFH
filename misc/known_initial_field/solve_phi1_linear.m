function [phi_1, q, out] = solve_phi1_linear(Delta_I, E_0, CS, lambda, maxit, tol)
% solve_phi1_linear
%
% Solves linearly for phi_1 using the first-order model
%   Delta_I(xi) ~= 2*Im( A(xi) * conj(B(xi)) )
% with
%   A(xi) = F{E_0}(xi)
%   B(xi) = F{ E_0(x) * phi_1(x) }(xi)
%
% Re-parameterization:
%   q(x) = A_0(x) * phi_1(x), real-valued on CS
%   E_0(x) = A_0(x) * exp(i*phi_0(x))
% so
%   E_0(x)*phi_1(x) = q(x) * exp(i*phi_0(x))
%
% Inputs:
%   Delta_I : NxN real differential focal-plane intensity
%   E_0     : NxN complex baseline pupil field
%   CS      : NxN logical compact-support mask
%   lambda  : Tikhonov regularization weight
%   maxit   : LSQR max iterations
%   tol     : LSQR tolerance
%
% Outputs:
%   phi_1   : NxN recovered drift function
%   q       : NxN recovered q = A_0 .* phi_1
%   out     : diagnostics struct

if nargin < 4 || isempty(lambda), lambda = 1e-6; end
if nargin < 5 || isempty(maxit),  maxit  = 50;   end
if nargin < 6 || isempty(tol),    tol    = 1e-6; end

N = size(E_0,1);
CS = logical(CS);
idxCS = find(CS);
N_CS = numel(idxCS);
Npix = N * N;

A_0 = abs(E_0);
phi_0 = angle(E_0);

P = zeros(N,N);
P(CS) = exp(1i * phi_0(CS));

A = F(E_0);

b = [Delta_I(:); zeros(N_CS,1)];


% Debug : Adjoint test
% lambda = 0;
% idxCS = find(CS);
% N_CS = numel(idxCS);
% Npix = N*N;
% 
% A = fftshift(fft2(ifftshift(E_0)));
% P = zeros(N,N);
% P(CS) = exp(1i * angle(E_0(CS)));
% 
% x = randn(N_CS,1);
% y = randn(Npix + N_CS,1);
% 
% Hx  = apply_augmented_operator(x,'notransp',A,P,idxCS,lambda,N,Npix,N_CS);
% HTy = apply_augmented_operator(y,'transp',  A,P,idxCS,lambda,N,Npix,N_CS);
% 
% lhs = x(:).' * HTy(:);
% rhs = Hx(:).' * y(:);
% 
% fprintf('adjoint test rel err = %.3e\n', abs(lhs-rhs)/max([1,abs(lhs),abs(rhs)]));

%


afun = @(x,transp_flag) apply_augmented_operator(x, transp_flag, ...
    A, P, idxCS, lambda, N, Npix, N_CS);

[x, flag, relres, iter, resvec] = lsqr(afun, b, tol, maxit);

q = zeros(N,N);
q(idxCS) = x;

phi_1 = zeros(N,N);
good = CS & (A_0 > 1e-12);
phi_1(good) = q(good) ./ A_0(good);

out.flag = flag;
out.relres = relres;
out.iter = iter;
out.resvec = resvec;
out.N_CS = N_CS;
out.lambda = lambda;

end

function y = apply_augmented_operator(x, transp_flag, A, P, idxCS, lambda, N, Npix, N_CS)

switch transp_flag
    case 'notransp'
        q = zeros(N,N);
        q(idxCS) = x;

        u = F(P .* q);
        y1 = 2 * imag(A .* conj(u));
        y2 = sqrt(lambda) * x;

        y = [y1(:); y2];

    case 'transp'
        eta = reshape(x(1:Npix), N, N);
        z   = x(Npix+1:end);

        tmp = FH((-2i * A) .* eta);
        q_adj_grid = real(conj(P) .* tmp);

        y = q_adj_grid(idxCS) + sqrt(lambda) * z;

    otherwise
        error('transp_flag must be ''notransp'' or ''transp''');
end

end


function U = F(u)
U = fftshift(fft2(ifftshift(u)));
end

function u = FH(U)
u = fftshift(ifft2(ifftshift(U))) * numel(U);
end