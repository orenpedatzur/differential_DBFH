function [out, seg_id] = solve_sign_problem_with_segments(absz, rez, support_mask, varargin)

% --- parse args (same as before) ---
p = inputParser;
p.addParameter('SmoothSigma', 0.5);
p.addParameter('QThresholdRel', 1e-4);
p.addParameter('MinSegmentPixels', 1);

p.addParameter('Tol', 1e-10);
p.addParameter('MaxIter', 1000);
p.addParameter('LambdaSeg', 1);
p.addParameter('Verbose', true);
p.addParameter('UseSingle', false);
p.addParameter('ShowPlots', false);
p.parse(varargin{:});
args = p.Results;


% ---- build abs(q = Im(z)) ----
q2 = absz.^2 - rez.^2;
q2(q2 < 0) = 0; % can only help to set as zero as this is closer to reality than a negative value

% q2(q2 < 0 & q2 > -1e-15 * max(1, max(absz(:)).^2)) = 0;
% if any(q2(:) < 0)
%     error('|z|^2 - Re(z)^2 is negative beyond numerical tolerance.');
% end
q = sqrt(q2);

% --- segmentation ---
seg_id = build_constant_sign_segments(q2, ...
    'SmoothSigma', args.SmoothSigma, ...
    'QThresholdRel', args.QThresholdRel, ...
    'MinSegmentPixels', args.MinSegmentPixels,...
    'ShowPlots', args.ShowPlots);

% --- remove the boundary segment ---
if any(seg_id(:) > 0)
    boundary_label = max(seg_id(:));
    seg_id(seg_id == boundary_label) = 0;
    seg_id = relabel_segments(seg_id);
end

% --- solve ---
out = solve_sign_problem_linear_2d_with_segments( ...
    absz, rez, support_mask, seg_id, ...
    'Tol', args.Tol, ...
    'MaxIter', args.MaxIter, ...
    'LambdaSeg', args.LambdaSeg, ...
    'Verbose', args.Verbose, ...
    'UseSingle', args.UseSingle,...
    'ShowPlots', args.ShowPlots);

% optional: attach q for consistency check
out.q_from_seg = q;
end