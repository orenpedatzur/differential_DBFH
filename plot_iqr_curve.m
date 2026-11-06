function h = plot_iqr_curve(x, med, q25, q75, lineSpec, xq25, xq75)
% Vertical phase IQR and horizontal SNR IQR; marker at mean SNR, median phase.
% Draw horizontal intervals at their actual endpoints: the mean can lie
% OUTSIDE the IQR, so clipping negative error lengths would be incorrect.
% Five-argument calls retain vertical-only error bars.
% Optional trailing xq25,xq75 add horizontal quartile whiskers.
if nargin < 6, xq25 = []; end
if nargin < 7, xq75 = []; end
assert(isempty(xq25)==isempty(xq75), 'Supply both x quartiles or neither.');
showX = ~isempty(xq25);
x=x(:); med=med(:); q25=q25(:); q75=q75(:);
assert(numel(x)==numel(med) && numel(x)==numel(q25) && ...
    numel(x)==numel(q75), 'x and phase arrays must have equal lengths.');
valid = isfinite(x) & isfinite(med) & isfinite(q25) & isfinite(q75);
if showX
    xq25=xq25(:); xq75=xq75(:);
    assert(numel(xq25)==numel(x) && numel(xq75)==numel(x), ...
        'X quartiles must have the same length as x.');
    valid = valid & isfinite(xq25) & isfinite(xq75);
    xq25=xq25(valid); xq75=xq75(valid);
    assert(all(xq25<=xq75), 'Invalid x quartile ordering.');
end
% Preserve original vertical-error behavior, including its eps floor.
x=x(valid); med=max(med(valid),eps);
q25=max(q25(valid),eps); q75=max(q75(valid),eps);
lo=max(med-q25,0); hi=max(q75-med,0);
h = errorbar(x,med,lo,hi,lineSpec, ...
    'LineWidth',2.5,'CapSize',5,'MarkerSize',8);
if ~showX, return; end
color = h.Color;
ax = ancestor(h,'axes');
% Extra whisker objects must not consume colors intended for later curves.
nextColor = ax.ColorOrderIndex;
nextStyle = ax.LineStyleOrderIndex;
capFactor = 10^0.008; % Symmetric cap height in logarithmic y coordinates.
for i = 1:numel(x)
    line([xq25(i) xq75(i)],[med(i) med(i)], ...
        'Color',color,'LineWidth',h.LineWidth,'HandleVisibility','off');
    line([xq25(i) xq25(i)],[med(i)/capFactor med(i)*capFactor], ...
        'Color',color,'LineWidth',h.LineWidth,'HandleVisibility','off');
    line([xq75(i) xq75(i)],[med(i)/capFactor med(i)*capFactor], ...
        'Color',color,'LineWidth',h.LineWidth,'HandleVisibility','off');
end
ax.ColorOrderIndex = nextColor;
ax.LineStyleOrderIndex = nextStyle;
end
