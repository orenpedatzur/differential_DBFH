function seg_id = build_constant_sign_segments(q, varargin)
p = inputParser;
p.addParameter('SmoothSigma', 1.0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('QThresholdRel', 0.05, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('MinSegmentPixels', 8, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('ShowPlots', false, @(x) islogical(x)|| isnumeric(x));
p.parse(varargin{:});

sigma = p.Results.SmoothSigma;
qThrRel = p.Results.QThresholdRel;
minSegPixels = p.Results.MinSegmentPixels;
show_plots = p.Results.ShowPlots;

q = double(q);
q = q / max(q(:) + eps);

if sigma > 0
    qs = imgaussfilt(q, sigma);
else
    qs = q;
end

valid = qs > qThrRel;

L = watershed(-qs);
L(~valid) = 0;

% Watershed ridge pixels inside the valid region = boundary group
boundary_mask = valid & (L == 0);

% Only non-boundary pixels are segmented here
mask = valid & (L > 0);
CC = bwconncomp(mask, 4);

seg_id = zeros(size(q), 'uint32');
next_id = uint32(0);

for k = 1:CC.NumObjects
    pix = CC.PixelIdxList{k};
    if numel(pix) < minSegPixels
        continue;
    end
    next_id = next_id + 1;
    seg_id(pix) = next_id;
end

leftover = mask & seg_id == 0;
CC2 = bwconncomp(leftover, 4);
for k = 1:CC2.NumObjects
    pix = CC2.PixelIdxList{k};
    if numel(pix) < minSegPixels
        continue;
    end
    next_id = next_id + 1;
    seg_id(pix) = next_id;
end

seg_id = relabel_segments(seg_id);

% make boundary thicker to avoid resolution misdetection problems
% boundary_mask = bwmorph(boundary_mask,'thicken',1);

% Boundary always gets the largest positive label
if any(boundary_mask(:))
    seg_id(boundary_mask) = 0;
end

if show_plots
    figure;
    seg_mask = ones(size(q));
    seg_mask(seg_id==0) = nan;
    ax2 = subplot(1,2,2);imagesc(seg_mask.*log(q));
    ax1 = subplot(1,2,1);imagesc(log(q));
    linkaxes([ax1,ax2]);sgtitle('segmentation of Im(z) for sign problem');
end

end