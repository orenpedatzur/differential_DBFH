function [snrValue, mask, info] = psfSNR(image, energyPercent, includePoisson, edgePixels, smoothSigma, show_plots)
%PSFSNR Energy-mask SNR of ONE full-aperture PSF image.
% SNR = psfSNR(image, energyPercent, includePoisson, edgePixels)
% [SNR, mask, info] = psfSNR(...) returns the mask and diagnostics too.
%
% image: real 2-D image. With includePoisson=true, values must be accumulated
%   detected photoelectron counts (not ADU, normalized intensity or an average
%   of multiple exposures). Gaussian-only images may use arbitrary units.
% energyPercent: percentage in (0,100], e.g. 50 for 50% of PSF energy.
% includePoisson: logical true/false, enabling source shot-noise variance.
% edgePixels: positive integer border width on EACH of the four sides.
% show_plots: optional logical flag (default false). Shows the log-scaled
%   background-subtracted PSF with a cyan signal mask and orange noise border.
%
% The border mean is subtracted from the entire image. The sample variance
% of the border estimates spatially uniform per-pixel background variance.
% Corners are counted once. The border must be uniform and signal-free.
%
% smoothSigma: optional Gaussian STD in pixels (default 1); zero disables
%   smoothing. Uses separable convolution with symmetric edge padding;
%   no Image Processing Toolbox is required.
%
% The energy target is energyPercent/100 times the SIGNED sum of the
% background-subtracted original image. A Gaussian-smoothed copy is ranked
% brightest first, and its cumulative signal determines the mask. No negative
% values are clipped in either the energy target or signal calculation.
% Smoothing is used ONLY for mask selection, never for noise estimation.
% The mask need not be connected or circular. Ties use MATLAB sort order.
%
% S = sum of background-subtracted counts in mask; n = nnz(mask).
% SNR = S / sqrt(n*sigmaBG^2 + includePoisson*S).
% Assumes independent pixels and negligible background-level estimation
% uncertainty. Background shot noise is already included in sigmaBG^2.
% No images are combined and no temporal variance is calculated.
%
% Smoothing reduces but does not eliminate data-dependent selection bias.
% The signed total is sensitive to background-offset error over large fields.
% The fraction in the original image can differ from the smoothed fraction.
%
% Example: [SNR, mask, info] = psfSNR(fullPSF, 50, true, 10);
% With plot and sigma=2 px: [SNR, mask, info] = psfSNR(fullPSF, 50, true, 10, true, 2);

if nargin < 5
    smoothSigma = 1;
end
if nargin < 6
    show_plots = false;
end

validateattributes(smoothSigma, {'numeric'}, ...
    {'real','finite','scalar','nonnegative'}, mfilename, 'smoothSigma', 6);
validateattributes(show_plots, {'logical'}, {'scalar'}, ...
    mfilename, 'show_plots', 5);

validateattributes(image, {'numeric'}, ...
    {'real','finite','nonempty','nonsparse','2d'}, mfilename, 'image', 1);
validateattributes(energyPercent, {'numeric'}, ...
    {'real','finite','scalar','>',0,'<=',100}, mfilename, 'energyPercent', 2);
validateattributes(includePoisson, {'logical'}, {'scalar'}, ...
    mfilename, 'includePoisson', 3);
validateattributes(edgePixels, {'numeric'}, ...
    {'real','finite','scalar','integer','positive'}, mfilename, 'edgePixels', 4);

[h,w] = size(image);
if 2*edgePixels >= min(h,w)
    error('psfSNR:Border', 'Border width must leave a nonempty interior.');
end
border = false(h,w);
border(1:edgePixels,:) = true;
border(end-edgePixels+1:end,:) = true;
border(:,1:edgePixels) = true;
border(:,end-edgePixels+1:end) = true;

image = double(image);
bgMean = mean(image(border));
bgVariance = var(image(border),0);
corrected = image - bgMean;
totalSignal = sum(corrected(:));
if totalSignal <= 0
    error('psfSNR:NoSignal', 'Signed total signal must be positive after background subtraction.');
end
smoothed = corrected;
if smoothSigma > 0
    radius = ceil(4*smoothSigma);
    x = -radius:radius;
    kernel = exp(-0.5*(x/smoothSigma).^2);
    kernel = kernel/sum(kernel);
    % Symmetric padding including the boundary pixel; supports small images.
    rows = mod((1-radius:h+radius)-1,2*h)+1;
    rows = min(rows,2*h+1-rows);
    cols = mod((1-radius:w+radius)-1,2*w)+1;
    cols = min(cols,2*w+1-cols);
    smoothed = conv2(kernel(:),kernel(:).',corrected(rows,cols),'valid');
end
[sorted, order] = sort(smoothed(:),'descend');
cumulative = cumsum(sorted);
targetSignal = (energyPercent/100)*totalSignal;
nKeep = find(cumulative >= targetSignal,1);
if isempty(nKeep)
    % Permit only roundoff-sized discrepancies (e.g. a 100% target).
    [peakCumulative,nKeep] = max(cumulative);
    if targetSignal-peakCumulative > 100*eps(max(1,sum(abs(smoothed(:)))))
        error('psfSNR:MaskTarget', 'Smoothed image cannot reach the requested signal target.');
    end
end
mask = false(h,w);
mask(order(1:nKeep)) = true;

signal = sum(corrected(mask));
if signal <= 0
    error('psfSNR:MaskSignal', 'Original-image signal inside the selected mask is not positive.');
end
backgroundVariance = nKeep*bgVariance;
shotVariance = double(includePoisson)*signal;
noiseSTD = sqrt(backgroundVariance + shotVariance);
snrValue = signal/noiseSTD; % Positive signal / zero noise correctly gives Inf.

info.signal = signal;
info.noiseSTD = noiseSTD;
info.backgroundMean = bgMean;
info.backgroundSTD = sqrt(bgVariance);
info.backgroundVariance = backgroundVariance; % Total within mask.
info.shotVariance = shotVariance;
info.nMaskPixels = nKeep;
info.totalSignal = totalSignal;
info.targetSignal = targetSignal;
info.enclosedFraction = sum(smoothed(mask))/totalSignal;
info.originalEnclosedFraction = signal/totalSignal;
info.smoothSigma = smoothSigma;
info.smoothedImage = smoothed;
info.edgeMask = border;
info.backgroundSubtractedImage = corrected;

if show_plots
    % Log display reveals tails; its floor does not affect SNR calculations.
    relativePSF = max(corrected,0)/max(corrected(:));
    fig = figure('Name','PSF and SNR mask','Color','w');
    ax = axes('Parent',fig);
    imagesc(ax,log10(max(relativePSF,1e-6)),[-6 0]);
    axis(ax,'image');
    colormap(ax,gray(256));
    cb = colorbar(ax);
    cb.Label.String = 'log_{10}(signal / peak), display floor = -6';
    hold(ax,'on');
    % Truecolor overlay leaves the grayscale PSF color scale unchanged.
    overlay = zeros(h,w,3);
    overlay(:,:,2) = 1;
    overlay(:,:,3) = 1;
    overlayHandle = imagesc(ax,overlay);
    set(overlayHandle,'AlphaData',0.35*double(mask));
    borderOverlay = zeros(h,w,3);
    borderOverlay(:,:,1) = 1;
    borderOverlay(:,:,2) = 0.5;
    borderHandle = imagesc(ax,borderOverlay);
    set(borderHandle,'AlphaData',0.35*double(border));
    % Proxy markers provide an explicit key for the two truecolor overlays.
    signalKey = plot(ax,NaN,NaN,'s','MarkerFaceColor',[0 1 1], ...
        'MarkerEdgeColor','none','DisplayName','Signal mask');
    borderKey = plot(ax,NaN,NaN,'s','MarkerFaceColor',[1 0.5 0], ...
        'MarkerEdgeColor','none','DisplayName','Background-noise border');
    legend(ax,[signalKey borderKey],'Location','southoutside');
    hold(ax,'off');
    xlabel(ax,'Column (pixel)');
    ylabel(ax,'Row (pixel)');
    title(ax,sprintf('SNR = %.4g | Cyan mask: %.2f%% smoothed energy, %d pixels', ...
        snrValue,100*info.enclosedFraction,nKeep));
end
end
