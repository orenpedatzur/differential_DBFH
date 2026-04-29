function Afun = make_Afun_lsqr_stacked_diagprecond(sys, p)
%MAKE_AFUN_LSQR_STACKED_DIAGPRECOND
% Real-stacked LSQR wrapper for right diagonal preconditioning.
%
% Physical unknown:
%   x = D u
%
% where D = diag(p), real positive.
%
% LSQR solves:
%   A D u = b
%
% After solve:
%   xstack = [p.*ur; p.*ui]

    n = sys.n;
    m = sys.m;

    p = p(:);
    assert(numel(p) == n, 'p must have length sys.n');
    assert(isnumeric(p), 'p must be numeric');
    assert(all(isfinite(p)), 'p contains NaN/Inf');

    Afun = @afun_local;

    function y = afun_local(x, transp_flag)

        switch transp_flag
            case 'notransp'
                ur = x(1:n);
                ui = x(n+1:end);

                % Right preconditioning: z = D*u
                z = p .* (ur + 1i*ui);

                Az = sys.Amul(z);
                y = [real(Az); imag(Az)];

            case 'transp'
                yr = x(1:m);
                yi = x(m+1:end);
                yc = yr + 1i*yi;

                g = sys.ATmul(yc);

                % Adjoint of A*D is D*A'
                g = p .* g;

                y = [real(g); imag(g)];

            otherwise
                error('Unknown transp_flag: %s', transp_flag);
        end
    end
end