function stats = diff_dbfh_solution_diagnostics(sys, x1vec, varargin)
%DIFF_DBFH_SOLUTION_DIAGNOSTICS  Basic checks for a differential-DBFH LSQR solution.
%
% Usage:
%   stats = diff_dbfh_solution_diagnostics(sys, x1vec);
%   stats = diff_dbfh_solution_diagnostics(sys, x1vec, X1_true);
%
% x1vec is the complex solution on sys.K, after undoing any right preconditioner.
% X1_true, if supplied, is a full HxW centered-grid phasor for A1.

    bnorm = max(norm(sys.b(:)), eps);

    r_raw = sys.Amul(x1vec(:)) - sys.b(:);
    out_raw = diff_dbfh_unpack_solution(sys, x1vec(:), 'normalizePhasor', false);
    out_norm = diff_dbfh_unpack_solution(sys, x1vec(:), 'normalizePhasor', true);

    stats = struct();
    stats.relres_raw_operator = norm(r_raw(:)) / bnorm;
    stats.leak1_raw = out_raw.leak1;
    stats.leak2_raw = out_raw.leak2;
    stats.leak1_normalized = out_norm.leak1;
    stats.leak2_normalized = out_norm.leak2;
    stats.abs_x_median = median(abs(x1vec(:)));
    stats.abs_x_min = min(abs(x1vec(:)));
    stats.abs_x_max = max(abs(x1vec(:)));
    stats.abs_x_p01 = prctile(abs(x1vec(:)), 1);
    stats.abs_x_p99 = prctile(abs(x1vec(:)), 99);

    fprintf('\n--- differential DBFH solution diagnostics ---\n');
    fprintf('operator relres, raw x       = %.3e\n', stats.relres_raw_operator);
    fprintf('raw leaks      leak1/leak2   = %.3e / %.3e\n', stats.leak1_raw, stats.leak2_raw);
    fprintf('normalized leaks leak1/leak2 = %.3e / %.3e\n', stats.leak1_normalized, stats.leak2_normalized);
    fprintf('|x| median/min/max           = %.3e / %.3e / %.3e\n', ...
        stats.abs_x_median, stats.abs_x_min, stats.abs_x_max);
    fprintf('|x| p01/p99                  = %.3e / %.3e\n', ...
        stats.abs_x_p01, stats.abs_x_p99);

    if nargin >= 3 && ~isempty(varargin{1})
        X1_true = varargin{1};
        assert(isequal(size(X1_true), size(sys.K)), 'X1_true must be full HxW.');
        assert(isfield(sys, 'anchor_idx'), 'sys.anchor_idx is required.');

        X1a = X1_true;
        anch = X1a(sys.anchor_idx);
        if abs(anch) == 0
            warning('True X1 is zero at sys.anchor_idx; cannot anchor-align true phasor.');
        else
            if isfield(sys, 'anchorVal')
                target = sys.anchorVal;
            else
                target = 1;
            end
            X1a = X1a * (target / anch);
            xtrue = X1a(sys.K);
            r_true = sys.Amul(xtrue(:)) - sys.b(:);
            out_true = diff_dbfh_unpack_solution(sys, xtrue(:), 'normalizePhasor', false);

            stats.relres_true_operator = norm(r_true(:)) / bnorm;
            stats.leak1_true = out_true.leak1;
            stats.leak2_true = out_true.leak2;

            fprintf('\n--- true phasor feasibility under current sys ---\n');
            fprintf('operator relres, true X1     = %.3e\n', stats.relres_true_operator);
            fprintf('true leaks     leak1/leak2   = %.3e / %.3e\n', ...
                stats.leak1_true, stats.leak2_true);
        end
    end
end
