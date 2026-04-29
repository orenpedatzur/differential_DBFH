function [Atilde_mul, Atilde_ATmul, Minv] = right_diag_precond_from_diag(sys, d)
% Build right-preconditioned operator using M = sqrt(d).
% Unknown transform: x = Minv .* u   where Minv = 1./M

    M = sqrt(d);                 % n-by-1 positive
    Minv = 1 ./ M;               % n-by-1

    Atilde_mul   = @(u) sys.Amul( Minv .* u );
    Atilde_ATmul = @(y) Minv .* sys.ATmul(y);  % since M real => M^{-H} = M^{-1}
end
