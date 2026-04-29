function [ImC, C, info] = raz_im_from_re_general(ReC, supp1, supp2)
% Recover Im(A1^* A2) from Re(A1^* A2) line-by-line using Raz 2014 logic,
% generalized to arbitrary NON-OVERLAPPING ordered supports on each x-line.
%
% INPUTS
%   ReC   : [Nx, Ny] real array. Column k is Re{A1^* A2} for line k.
%   supp1 : [Nx, Ny] logical support mask of E1 along x-lines
%   supp2 : [Nx, Ny] logical support mask of E2 along x-lines
%
% OUTPUTS
%   ImC   : [Nx, Ny] recovered imaginary part
%   C     : [Nx, Ny] complex cross-term = ReC + 1i*ImC
%   info  : diagnostics struct
%
% IMPORTANT
%   1) This assumes FFT/ IFFT are taken along x only, with no fftshift.
%   2) Each column of supp1 and supp2 must be a single interval.
%   3) The two intervals must be disjoint and ordered.
%   4) The implied support of ifft(C(:,k)) must be one-sided enough that
%      every unknown sample can be recovered from Hermitian symmetry.

    ReC   = double(ReC);
    supp1 = logical(supp1);
    supp2 = logical(supp2);

    [Nx, Ny] = size(ReC);
    assert(isequal(size(supp1), [Nx, Ny]), 'supp1 size mismatch');
    assert(isequal(size(supp2), [Nx, Ny]), 'supp2 size mismatch');

    ImC = nan(Nx, Ny);
    C   = nan(Nx, Ny);

    info.valid_line      = false(1, Ny);
    info.reason          = strings(1, Ny);
    info.interval1       = nan(Ny, 2);
    info.interval2       = nan(Ny, 2);
    info.support_q       = false(Nx, Ny);
    info.rel_leak        = nan(1, Ny);

    for ky = 1:Ny
        m1 = supp1(:, ky);
        m2 = supp2(:, ky);

        % ---- validate each support is a single interval
        [ok1, a1, b1] = single_interval(m1);
        [ok2, a2, b2] = single_interval(m2);

        if ~ok1 || ~ok2
            info.reason(ky) = "support is not a single interval";
            continue;
        end

        info.interval1(ky,:) = [a1, b1];
        info.interval2(ky,:) = [a2, b2];

        % ---- validate non-overlap + orderedness
        if any(m1 & m2)
            info.reason(ky) = "supports overlap";
            continue;
        end

        ordered = (b1 < a2) || (b2 < a1);
        if ~ordered
            info.reason(ky) = "supports are not ordered";
            continue;
        end

        % ---- determine support of q = ifft(C) directly from supports
        % C = A1 .* conj(A2), q = ifft(C)
        % support depends only on support geometry, so use ones.
        qtest = ifft(fft(double(m1)) .* conj(fft(double(m2))));
        sq = abs(qtest) > 1e-12;
        info.support_q(:, ky) = sq;

        % must be one contiguous interval
        [okq, aq, bq] = single_interval(sq);
        if ~okq
            info.reason(ky) = "ifft cross-term support is not a single interval";
            continue;
        end

        % ---- Raz step 1: inverse FFT of the known real part
        R = ifft(ReC(:, ky));

        % q = R + 1i*Ispace must vanish outside sq
        % => Ispace(outside) = 1i * R(outside)
        Ispace = nan(Nx,1);
        outside = ~sq;
        Ispace(outside) = 1i * R(outside);

        % ---- Raz step 2: use Hermitian symmetry of Ispace because ImC is real
        % If ImC is real, then ifft(ImC) is Hermitian:
        % Ispace(j) = conj(Ispace(mirror(j)))
        unresolved = find(sq);
        for jj = unresolved(:).'
            jm = mirror_idx(jj, Nx);

            if outside(jm)
                Ispace(jj) = conj(Ispace(jm));
            end
        end

        if any(isnan(Ispace))
            info.reason(ky) = "causality insufficient: some inside samples have no mirrored outside sample";
            continue;
        end

        % ---- back to Fourier domain
        ImC(:, ky) = real(fft(Ispace));
        C(:, ky)   = ReC(:, ky) + 1i * ImC(:, ky);

        % ---- diagnostic: leakage outside inferred support
        q = ifft(C(:, ky));
        info.rel_leak(ky) = norm(q(outside)) / max(norm(q), eps);
        info.valid_line(ky) = true;
        info.reason(ky) = "ok";
    end
end


function [ok, a, b] = single_interval(mask)
    idx = find(mask(:).');
    if isempty(idx)
        ok = false; a = nan; b = nan; return;
    end
    a = idx(1);
    b = idx(end);
    ok = all(mask(a:b)) && (nnz(mask) == (b-a+1));
end


function jm = mirror_idx(j, N)
    % Hermitian mirror index for MATLAB ifft/fft with no shift
    % 1 -> 1, 2 -> N, 3 -> N-1, ..., N -> 2
    if j == 1
        jm = 1;
    else
        jm = N - j + 2;
    end
end