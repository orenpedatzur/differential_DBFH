function d = hutchinson_diag_AtA(sys, nsamp, blockSize, useGPU)
% Estimate diag(A'*A) for complex operator A via Hutchinson probing.
% sys.Amul  : z -> A*z   (complex)
% sys.ATmul : y -> A'*y  (complex)
%
% d is n-by-1, real, >=0

    if nargin < 2 || isempty(nsamp),     nsamp = 20; end
    if nargin < 3 || isempty(blockSize), blockSize = 1; end
    if nargin < 4,                      useGPU = false; end

    n = sys.n;
    d = zeros(n,1,'double');

    if useGPU
        d = gpuArray(d);
    end

    % We'll accumulate in blocks to reduce overhead (optional)
    k = 0;
    while k < nsamp
        bs = min(blockSize, nsamp - k);

        % Complex unit-circle probes: v_i = exp(1j*theta)
        if useGPU
            theta = 2*pi*rand(n, bs, 'like', d);      % gpu
        else
            theta = 2*pi*rand(n, bs);
        end
        V = exp(1i*theta);                             % n x bs

        for j = 1:bs
            v = V(:,j);
            Av = sys.Amul(v);
            AtAv = sys.ATmul(Av);
            % Hutchinson diagonal estimator for complex Hermitian:
            d = d + real(conj(v) .* AtAv);
        end

        k = k + bs;
    end

    d = d / nsamp;

    % Numerical safety: clamp tiny/negative (can happen from finite samples)
    d = max(d, 1e-30);

    if useGPU
        d = gather(d);
    end
end
