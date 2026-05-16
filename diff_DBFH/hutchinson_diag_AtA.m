function d = hutchinson_diag_AtA(sys, nsamp, blockSize, useGPU)
% Estimate diag(A'*A) for complex operator A using complex unit probes.
% GPU-clean: no gather/double inside. Returned d stays on same device as sys.b.

    if nargin < 2 || isempty(nsamp),     nsamp = 20; end
    if nargin < 3 || isempty(blockSize), blockSize = 1; end
    if nargin < 4 || isempty(useGPU)
        useGPU = isa(sys.b, 'gpuArray');
    end

    n = sys.n;

    if useGPU
        d = gpuArray.zeros(n,1);
    else
        d = zeros(n,1);
    end

    k = 0;
    while k < nsamp
        bs = min(blockSize, nsamp - k);

        if useGPU
            theta = 2*pi*rand(n, bs, 'gpuArray');
        else
            theta = 2*pi*rand(n, bs);
        end

        V = exp(1i*theta);

        for j = 1:bs
            v = V(:,j);
            Av = sys.Amul(v);
            AtAv = sys.ATmul(Av);
            d = d + real(conj(v) .* AtAv);
        end

        k = k + bs;
    end

    d = d / nsamp;

    % Do not use exact zeros. They create absurd preconditioners.
    d = max(real(d), 1e-30);
end