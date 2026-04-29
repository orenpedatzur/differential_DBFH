function [adj_err, true_relres] = test_stacked_operator(Afun, sys, xtrue, p, bstack)
%TEST_STACKED_OPERATOR Check preconditioned real-stacked feasibility and adjointness.
% Afun represents A*D, where D = diag(p). Therefore the true
% preconditioned variable is u_true = D^{-1} x_true.

assert(numel(xtrue) == sys.n, 'xtrue must have length sys.n.');
utrue = xtrue ./ p(:);
utrue_stack = [real(utrue); imag(utrue)];
rstack = Afun(utrue_stack, 'notransp') - bstack;
true_relres = norm(rstack) / max(norm(bstack), eps);

x = randn(2*sys.n,1);
y = randn(2*sys.m,1);

Ax  = Afun(x, 'notransp');
ATy = Afun(y, 'transp');

lhs = y(:)' * Ax(:);
rhs = ATy(:)' * x(:);
adj_err = abs(lhs-rhs) / max(abs(lhs), abs(rhs));

end