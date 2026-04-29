function Afun = make_Afun_lsqr_stacked(sys, Minv)
% Afun(x,'notransp') returns [real(A*x); imag(A*x)]
% Afun(y,'transp')   returns [real(A'*y); imag(A'*y)]
%
% If Minv provided (n-by-1), we solve A*(Minv.*u)=b in variable u.

    if nargin < 2 || isempty(Minv)
        mul   = @(z) sys.Amul(z);
        atmul = @(y) sys.ATmul(y);
    else
        mul   = @(u) sys.Amul(Minv .* u);
        atmul = @(y) Minv .* sys.ATmul(y);
    end

    n = sys.n;
    m = sys.m;

    Afun = @(x,transpFlag) local_apply(x, transpFlag, mul, atmul, n, m);
end

function out = local_apply(x, transpFlag, mul, atmul, n, m)
    switch transpFlag
        case 'notransp'
            % x is 2n real stacked => complex vector
            zr = x(1:n);
            zi = x(n+1:2*n);
            z  = complex(zr, zi);

            y = mul(z);  % complex m-by-1
            out = [real(y); imag(y)];

        case 'transp'
            % x is 2m real stacked => complex vector
            yr = x(1:m);
            yi = x(m+1:2*m);
            y  = complex(yr, yi);

            z = atmul(y); % complex n-by-1
            out = [real(z); imag(z)];

        otherwise
            error('transpFlag must be ''notransp'' or ''transp''.');
    end
end
