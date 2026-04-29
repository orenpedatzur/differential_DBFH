function Afun = make_Afun_lsqr_stacked_no_precond(sys)
%MAKE_AFUN_LSQR_STACKED_NO_PRECOND
% Real-stacked LSQR wrapper for complex system sys.Amul/sys.ATmul.
%
% Complex system:
%   A z = b
%
% Real-stacked system:
%   [real(Az); imag(Az)] = [real(b); imag(b)]
%
% Unknown:
%   xstack = [real(z); imag(z)]

    n = sys.n;
    m = sys.m;

    Afun = @afun_local;

    function y = afun_local(x, transp_flag)

        switch transp_flag
            case 'notransp'
                xr = x(1:n);
                xi = x(n+1:end);
                z = xr + 1i*xi;

                Az = sys.Amul(z);
                y = [real(Az); imag(Az)];

            case 'transp'
                yr = x(1:m);
                yi = x(m+1:end);
                yc = yr + 1i*yi;

                g = sys.ATmul(yc);
                y = [real(g); imag(g)];

            otherwise
                error('Unknown transp_flag: %s', transp_flag);
        end
    end
end