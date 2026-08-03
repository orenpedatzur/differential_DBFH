close all; clear all;clc; rng(1);

addpath('ImZ_from_ReZ');
addpath('diff_DBFH');
addpath('fftc');

show_plots = true;

%% ========================================================================
%  USER PARAMETERS
%  ========================================================================

zernike_nMax            = 2;
zernike_coeff_sigma_rad = 0.5;

% ------------------------------------------------------------------------
% Physical telescope / detector parameters
% ------------------------------------------------------------------------
% The sampling is set in the physically correct order:
%   1) choose telescope diameter, focal length, wavelength, detector pitch;
%   2) derive the pupil-plane FFT window L = lambda*f/p_cam;
%   3) derive the pupil-plane pixel size dx = L/N;
%   4) derive the aperture diameter and segment size in pupil pixels.
%
% This replaces the older logic where seg_radius_pix was chosen first and
% the focal-plane pixel pitch was only inferred afterward.

N                 = 2048;       % FFT grid size; also the modeled detector size [pixels]
D_m_inscribed     = 8.2;        % HWO EAC5 inscribed aperture diameter [m]
f_m               = 171;        % HWO effective focal length [m]
lambda_m          = 550e-9;     % reference wavelength [m]
cam_px_m          = 3.45e-6;     % detector pixel pitch [m]
num_rings         = 3;          % 3 rings -> 37 hex segments

% Focal-plane sampling is fixed by the camera.
px_m_focal_plane = cam_px_m;

% Pupil-plane FFT window required to give this focal-plane sampling.
L_m_pupil        = lambda_m * f_m / px_m_focal_plane;
px_m_pupil_plane = L_m_pupil / N;

% Aperture diameter in the pupil-grid coordinates.
D_pix_inscribed = D_m_inscribed / px_m_pupil_plane;

aperture_diameter_factor = (2*num_rings + 1)*sqrt(3);


seg_radius_pix_exact = D_pix_inscribed / aperture_diameter_factor;
seg_radius_pix       = round(seg_radius_pix_exact);

% Recompute the effective physical sampling after rounding seg_radius_pix.
D_pix_inscribed_eff     = aperture_diameter_factor * seg_radius_pix;
px_m_pupil_plane_eff    = D_m_inscribed / D_pix_inscribed_eff;
L_m_pupil_eff           = N * px_m_pupil_plane_eff;
px_m_focal_plane_eff    = lambda_m * f_m / L_m_pupil_eff;
Q_pix_per_lambdaD_eff   = lambda_m * f_m / (D_m_inscribed * px_m_focal_plane_eff);
lambdaD_per_pix_eff     = 1 / Q_pix_per_lambdaD_eff;

% Perturbation geometry in pupil pixels.
% With physical sampling enabled, the old value string_depth_pix=10 is a
% much smaller fraction of a segment than it was for seg_radius_pix=40.
% Keep it fixed only if that is intentional.
string_depth_pix  = 25;
gap_pix           = 0;

% Backward-compatible variable names used later in the script.
Q = Q_pix_per_lambdaD_eff;

% Noise model: independent additive Gaussian noise on each PSF image,
% scaled by max(I_full(:)).
noise_level       = 1e-3;
use_noisy_mag1_for_dDBFH = true;

% Cross-term recovery controls for dDBFH.
d_theta           = 0.1;
theta_list        = 0:d_theta:180-d_theta;
use_exact_ReZ     = false;   % false: ReZ = DeltaI/2 approximation

% ---------------- BUILD SEGMENTED APERTURE ----------------
[seg_id, seg_centers, X, Y] = make_hex_segment_aperture(N, seg_radius_pix, num_rings);
aperture = seg_id > 0;
nSeg = max(seg_id(:));

% choose upper-most peripheral segment
[~, top_seg] = max(seg_centers(:,2));
top_mask = seg_id == top_seg;

% outward normal
c_top = seg_centers(top_seg,:);
n_out = c_top / norm(c_top);

S = X*n_out(1) + Y*n_out(2);
S_top = S(top_mask);
s_edge = max(S_top);

% string-cut cap
cut_E2 = s_edge - string_depth_pix;
cut_gap = cut_E2 - gap_pix;

E2_mask = top_mask & (S >= cut_E2);
gap_mask = top_mask & (S >= cut_gap) & (S < cut_E2);
E1_mask = aperture & ~E2_mask & ~gap_mask;

% ---------------- RANDOM SEGMENTED PHASE FRONT ----------------
E = make_random_segmented_field(seg_id, seg_centers, X, Y, seg_radius_pix, nSeg, zernike_nMax, zernike_coeff_sigma_rad);

E1 = E .* E1_mask;
E2 = E .* E2_mask;

% ---------------- FOCAL PLANE FIELDS ----------------
A1 = fft2c(E1);
A2 = fft2c(E2);
Atot = A1 + A2;

I_12 = abs(Atot).^2;
I_1  = abs(A1).^2;

Z = conj(A1) .* A2;
ReZ_exact = real(Z);
ImZ_true  = imag(Z);

DeltaI = I_12 - I_1;

if use_exact_ReZ
    ReZ_meas = ReZ_exact;
else
    % first-order approximation: neglect |A2|^2
    ReZ_meas = 0.5 * DeltaI;
end

% ---------------- SUPPORT OF g = IFT2[A1^* A2] ----------------
CSg = correlation_support(E1_mask, E2_mask);

% ---------------- KK / RADON RECOVERY OF ImZ ----------------
% [ImZ_raw, kk] = recover_ImZ_by_radon_KK(ReZ_meas, CSg, theta_list,'trueImZ',ImZ_true);
% ImZ_raw = circshift(ImZ_raw,[-1,-1]); % for some reason we end with a 1 px offset in every direction
[ImZ_raw, kk] = recover_ImZ_by_cartesian_KK(ReZ_meas, CSg, theta_list,'trueImZ', ImZ_true,'InterpMethod', 'cubic');


%%
fprintf('Line-domain gI error = %.3e\n', kk.line_rel_err);

% --- Cartesian line-domain truth check ---
if isfield(kk, 'gI_true_all') && ~isempty(kk.gI_true_all)

    gI_true = kk.gI_true_all;
    gI_rec  = kk.gI_all;

    line_err = norm(gI_rec(:) - gI_true(:)) / max(norm(gI_true(:)), eps);
    fprintf('Line-domain gI error = %.3e\n', line_err);

    ia = round(numel(theta_list)/4);

    slice_err = norm(gI_rec(:,ia) - gI_true(:,ia)) / max(norm(gI_true(:,ia)), eps);
    fprintf('Single-slice gI error at theta %.1f = %.3e\n', theta_list(ia), slice_err);

    figure('Color','w');
    subplot(2,2,1); plot(real(gI_true(:,ia))); title('real gI true');
    subplot(2,2,2); plot(real(gI_rec(:,ia)));  title('real gI rec');
    subplot(2,2,3); plot(imag(gI_true(:,ia))); title('imag gI true');
    subplot(2,2,4); plot(imag(gI_rec(:,ia)));  title('imag gI rec');

    figure('Color','w');
    subplot(2,2,1); imagesc(real(gI_true)); axis image; colorbar; title('Re[g_{\theta}] truth');
    subplot(2,2,2); imagesc(real(gI_rec));  axis image; colorbar; title('Re[g_{\theta}] rec');
    subplot(2,2,3); imagesc(imag(gI_true)); axis image; colorbar; title('Im[g_{\theta}] truth');
    subplot(2,2,4); imagesc(imag(gI_rec));  axis image; colorbar; title('Im[g_{\theta}] rec');

else
    warning('No true line-domain gI stored. Pass ''trueImZ'', ImZ_true into recover_ImZ_by_cartesian_KK.');
end

% ---------------- SUPPORT-BASED SCALE CORRECTION ----------------
% this tep can potentially help rescale the intensity in each pixel using
% the known Re and Im relationship outside the CSg (not only half plane)
gR = ifft2c(ReZ_meas);
gI_raw = ifft2c(ImZ_raw);

Mout = ~CSg;

u = 1i * gI_raw(Mout);
v = -gR(Mout);


gI_rec = ifft2c(ImZ_raw);
support_res = norm(gR(Mout) + 1i*gI_rec(Mout)) / max(norm(gR(Mout)), eps);


%%
f = figure('Color','w','Position',[100 100 2200 600]);

sub_fig_width  = 0.38;
sub_fig_height = 0.38;
x_offset = -0.07;

theta_display = 70; % degs
ind_theta_display = find(theta_display == theta_list);
M = tand(theta_list(ind_theta_display));
x_bottom = round(-N/2/M+N/2);
y_bottom = 1;
x_top = round(N/2/M+N/2);
y_top = N;

% --- styling ---
fs_title = 16;
fs_axis  = 16;
lw_plot  = 1.8;
arrow_lw = 2;

% zero-centered diverging colormap: blue -> white -> red
ncm  = 256;
cneg = [0.10 0.35 0.80];
cpos = [0.80 0.20 0.20];
cmap = [linspace(cneg(1),1,ncm/2)', linspace(cneg(2),1,ncm/2)', linspace(cneg(3),1,ncm/2)'; ...
        linspace(1,cpos(1),ncm/2)', linspace(1,cpos(2),ncm/2)', linspace(1,cpos(3),ncm/2)'];

% symmetric color limits so zero is white
cl_ReZ   = max(abs(ReZ_meas(:)));
cl_gR_re = max(abs(real(gR(:))));
cl_gR_im = max(abs(imag(gR(:))));
cl_gI_re = max(abs(real(gI_rec(:))));
cl_gI_im = max(abs(imag(gI_rec(:))));
cl_ImZ   = max(abs([ImZ_raw(:); ImZ_true(:)]));


% focal plane zoom-in coordinates
x_lim_focal_plane = N/2+N/50*[-1,1];
y_lim_focal_plane = N/2+N/50*[-1,1];

% -------------------- top row --------------------

ax1 = axes('Position',[1/6-sub_fig_width/2+x_offset, 0.5-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(ReZ_meas); axis xy image; box on;
colormap(ax1,cmap); clim(ax1,[-cl_ReZ cl_ReZ]); colorbar;
title('approx. Re(z)','FontSize',fs_title);
xlim(x_lim_focal_plane);ylim(y_lim_focal_plane);
xticks([]);yticks([]);
set(ax1,'FontSize',fs_axis,'LineWidth',1);

ax2 = axes('Position',[2/6-sub_fig_width/2+x_offset, 3/4-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(real(gR)); axis xy image; box on;
colormap(ax2,cmap); clim(ax2,[-cl_gR_re cl_gR_re]); colorbar;
title('Re(g_R(x,y))','FontSize',fs_title);
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax2,'FontSize',fs_axis,'LineWidth',1);

ax3 = axes('Position',[2/6-sub_fig_width/2+x_offset, 1/4-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(imag(gR)); axis xy image; box on;
colormap(ax3,cmap); clim(ax3,[-cl_gR_im cl_gR_im]); colorbar;
title('Im(g_R(x,y))','FontSize',fs_title);
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax3,'FontSize',fs_axis,'LineWidth',1);

ax4 = axes('Position',[3/6-sub_fig_width/2+x_offset, 3/4-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(real(kk.gR_all)); axis xy image; box on;
colormap(ax4,cmap); clim(ax4,[-cl_gR_re cl_gR_re]); colorbar;
title('Re(g_{R,\theta}(s)))','FontSize',fs_title);
hold on; xline(ind_theta_display,'--k');hold off;
text(ind_theta_display,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);
% rectangle('Position',[1,1,numel(theta_list),N/2],'Curvature',[0.2,0.4],'LineWidth',2,'LineStyle','--')
xlabel('\theta');ylabel('s');xticks([]);yticks([]);
set(ax4,'FontSize',fs_axis,'LineWidth',1);

ax5 = axes('Position',[3/6-sub_fig_width/2+x_offset, 1/4-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(imag(kk.gR_all)); axis xy image; box on;
colormap(ax5,cmap); clim(ax5,[-cl_gR_im cl_gR_im]); colorbar;
title('Im(g_{R,\theta}(s))','FontSize',fs_title);
hold on; xline(ind_theta_display,'--k');hold off;
text(ind_theta_display,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);
% rectangle('Position',[1,1,numel(theta_list),N/2],'Curvature',[0.2,0.4],'LineWidth',2,'LineStyle','--')
xlabel('\theta');ylabel('s');xticks([]);yticks([]);
set(ax5,'FontSize',fs_axis,'LineWidth',1);

ax6 = axes('Position',[4/6-sub_fig_width/2+x_offset, 3/4-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(real(kk.gI_all)); axis xy image; box on;
colormap(ax6,cmap); clim(ax6,[-cl_gR_re cl_gR_re]); colorbar;
title('Re(g_{I,\theta}(s)))','FontSize',fs_title);
hold on; xline(ind_theta_display,'--k');hold off;
text(ind_theta_display,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);
% rectangle('Position',[1,1,numel(theta_list),N/2],'Curvature',[0.2,0.4],'LineWidth',2,'LineStyle','--')
xlabel('\theta');ylabel('s');xticks([]);yticks([]);
set(ax6,'FontSize',fs_axis,'LineWidth',1);

ax7 = axes('Position',[4/6-sub_fig_width/2+x_offset, 1/4-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(imag(kk.gI_all)); axis xy image; box on;
colormap(ax7,cmap); clim(ax7,[-cl_gR_im cl_gR_im]); colorbar;
title('Im(g_{I,\theta}(s))','FontSize',fs_title);
hold on; xline(ind_theta_display,'--k');hold off;
text(ind_theta_display,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);
% rectangle('Position',[1,1,numel(theta_list),N/2],'Curvature',[0.2,0.4],'LineWidth',2,'LineStyle','--');
xlabel('\theta');ylabel('s');xticks([]);yticks([]);
set(ax7,'FontSize',fs_axis,'LineWidth',1);

ax8 = axes('Position',[5/6-sub_fig_width/2+x_offset, 3/4-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(real(kk.gI_cart)); axis xy image; box on;
colormap(ax8,cmap); clim(ax8,[-cl_gR_re cl_gR_re]); colorbar;
title('Re(g_I(x,y))','FontSize',fs_title);
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax8,'FontSize',fs_axis,'LineWidth',1);

ax9 = axes('Position',[5/6-sub_fig_width/2+x_offset, 1/4-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(imag(kk.gI_cart)); axis xy image; box on;
colormap(ax9,cmap); clim(ax9,[-cl_gR_im cl_gR_im]); colorbar;
title('Im(g_I(x,y)))','FontSize',fs_title);
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax9,'FontSize',fs_axis,'LineWidth',1);

ax10 = axes('Position',[6/6-sub_fig_width/2+x_offset, 0.5-sub_fig_height/2, sub_fig_width, sub_fig_height]);
imagesc(ImZ_raw); axis xy image; box on;
colormap(ax10,cmap); clim(ax10,[-cl_ReZ cl_ReZ]); colorbar;
title('recovered Im(z)','FontSize',fs_title);
xlim(x_lim_focal_plane);ylim(y_lim_focal_plane);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax10,'FontSize',fs_axis,'LineWidth',1);

% -------------------- arrows showing flow --------------------
% arrow_size = 0.025;
% arrow_x_loc = 1/6.1*[1:5];
% arrow_x_offset = 0*0.01;
% arrow_line_width = 8;


% annotation('arrow', arrow_x_loc(1)*[1,1]+[0,arrow_size]+arrow_x_offset, [0.5, 0.5], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.3*[1,1,1]);
% annotation('textbox', [arrow_x_loc(1)+0.005, 0.5+0.06, 0.01, 0.01],'String', 'FT^{-1}', ...
%     'EdgeColor', 'none','HorizontalAlignment', 'center','FontSize', 16,'FontWeight', 'bold');
% 
% annotation('arrow', arrow_x_loc(5)*[1,1]+[0,arrow_size]+arrow_x_offset, [0.5, 0.5], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.3*[1,1,1]);
% annotation('textbox', [arrow_x_loc(5)+0.005, 0.5+0.06, 0.01, 0.01],'String', 'FT', ...
%     'EdgeColor', 'none','HorizontalAlignment', 'center','FontSize', 16,'FontWeight', 'bold');
% 
% annotation('arrow', arrow_x_loc(2)*[1,1]+[0,arrow_size]+arrow_x_offset, [0.75, 0.75], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.3*[1,1,1]);
% annotation('textbox', [arrow_x_loc(2)+0.015, 0.75+0.06, 0.01, 0.01],'String', '(x,y)\rightarrow(s,\theta)', ...
%     'EdgeColor', 'none','HorizontalAlignment', 'center','FontSize', 12,'FontWeight', 'bold');
% 
% annotation('arrow', arrow_x_loc(2)*[1,1]+[0,arrow_size]+arrow_x_offset, [0.25, 0.25], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.3*[1,1,1]);
% 
% annotation('arrow', arrow_x_loc(4)*[1,1]+[0,arrow_size]+arrow_x_offset, [0.75, 0.75], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.3*[1,1,1]);
% 
% annotation('arrow', arrow_x_loc(4)*[1,1]+[0,arrow_size]+arrow_x_offset, [0.25, 0.25], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.3*[1,1,1]);
% 
% % cross arrows
% annotation('arrow', [0.46,0.54], [0.58, 0.26], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.5*[0,0,1]);
% 
% annotation('arrow', [0.46,0.54], [0.25, 0.58], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.5*[0,0,1]);
% 
% annotation('arrow', [0.62,0.62], [0.7, 0.8], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.5*[1,0,0]);
% 
% annotation('arrow', [0.62,0.62], [0.2, 0.3], ...
%     'LineWidth',arrow_line_width, 'HeadLength',20, 'HeadWidth',30,'color',0.5*[1,0,0]);

%%
%% Compare recovered and ground-truth g_I at one angle

theta_display = 70; % degrees
[~, ind_theta_display] = min(abs(theta_list - theta_display));
theta_used = theta_list(ind_theta_display);

s = kk.s;

gI_true_slice = kk.gI_true_all(:,ind_theta_display);
gI_rec_slice  = kk.gI_all(:,ind_theta_display);


cl_Re_gI = max(abs(kk.gI_cart(:)));


figure('Position',[100,100,2000,800]);
t = tiledlayout(2, 4, 'TileSpacing', 'tight', 'Padding', 'none');

ax(1) =  nexttile;
imagesc(real(kk.gI_cart)); hold on;
colormap(ax(1),cmap); clim(ax(1),[-cl_Re_gI cl_Re_gI]); colorbar;
title('recovered Re(g_I)','FontSize',fs_title);
% xlim(x_lim_focal_plane);ylim(y_lim_focal_plane);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax(1),'FontSize',fs_axis,'LineWidth',1.5);
axis xy image;
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);

ax(2) =  nexttile;
imagesc(real(ifft2c(ImZ_true))); hold on;
colormap(ax(2),cmap); clim(ax(2),[-cl_Re_gI cl_Re_gI]); colorbar;
title('truth Re(g_I)','FontSize',fs_title);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax(2),'FontSize',fs_axis,'LineWidth',1.5);
axis xy image;
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);

ax(3) =  nexttile;
imagesc(real(kk.gI_cart-ifft2c(ImZ_true))); hold on;
colormap(ax(3),cmap); clim(ax(3),[-cl_Re_gI cl_Re_gI]); colorbar;
title('recovered - truth','FontSize',fs_title);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax(3),'FontSize',fs_axis,'LineWidth',1.5);
axis xy image;
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);

ax(4) =  nexttile;
plot(s, real(gI_rec_slice),  'k-',  'LineWidth',1.5); hold on;
plot(s, real(gI_true_slice), 'r--', 'LineWidth',1.0);
grid on; box on;
xlabel('s');
ylabel('Re(g_I)');
legend(['Re(g_{I,\theta=',num2str(theta_display),'}(s)) reconstructed'],['Re(g_{I,\theta=',num2str(theta_display),'}(s)) truth'],'Location','southwest','Box','off','FontSize',fs_axis);
xticks([-1000:500:1000]);yticks([-0.5:0.1:0.5]);
ylim(1.4*max(abs(gI_rec_slice))*[-1,1]);
set(ax(4),'FontSize',fs_axis,'LineWidth',1.5);

ax(5) =  nexttile;
imagesc(imag(kk.gI_cart)); hold on;
colormap(ax(5),cmap); clim(ax(5),[-cl_Re_gI cl_Re_gI]); colorbar;
title('recovered Im(g_I)','FontSize',fs_title);
% xlim(x_lim_focal_plane);ylim(y_lim_focal_plane);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax(5),'FontSize',fs_axis,'LineWidth',1.5);
axis xy image;
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);

ax(6) =  nexttile;
imagesc(imag(ifft2c(ImZ_true))); hold on;
colormap(ax(6),cmap); clim(ax(6),[-cl_Re_gI cl_Re_gI]); colorbar;
title('truth Im(g_I)','FontSize',fs_title);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax(6),'FontSize',fs_axis,'LineWidth',1.5);
axis xy image;
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);

ax(7) =  nexttile;
imagesc(imag(kk.gI_cart-ifft2c(ImZ_true))); hold on;
colormap(ax(7),cmap); clim(ax(7),[-cl_Re_gI cl_Re_gI]); colorbar;
title('recovered - truth','FontSize',fs_title);
xlabel('x');ylabel('y');xticks([]);yticks([]);
set(ax(7),'FontSize',fs_axis,'LineWidth',1.5);
axis xy image;
hold on; plot([x_bottom,x_top],[y_bottom,y_top],'--k');hold off;
text(x_top,0.95*N,[' \theta=',num2str(theta_list(ind_theta_display)),'^o']);

ax(8) =  nexttile;
plot(s, imag(gI_rec_slice),  'k-',  'LineWidth',1.5); hold on;
plot(s, imag(gI_true_slice), 'r--', 'LineWidth',1.0);
grid on; box on;
xlabel('s');
ylabel('Im(g_I)');
legend(['Im(g_{I,\theta=',num2str(theta_display),'}(s)) reconstructed'],['Im(g_{I,\theta=',num2str(theta_display),'}(s)) truth'],'Location','northwest','Box','off','FontSize',fs_axis);
xticks([-1000:500:1000]);yticks([-0.5:0.1:0.5]);
ylim(1.4*max(abs(gI_rec_slice))*[-1,1]);
set(ax(8),'FontSize',fs_axis,'LineWidth',1.5);

linkaxes(ax([4,8]),'x');xlim([min(s),max(s)]);

%%



% ========================================================================
% SUPPORT OF g = IFT[A1^* A2]
% support is all displacement vectors x2 - x1
% ========================================================================
function CSg = correlation_support(E1_mask, E2_mask)

big = conv2(double(flipud(fliplr(E1_mask))), double(E2_mask), 'same');
CSg = big > 0;

end


% ========================================================================
% HWO / JWST-LIKE HEX SEGMENTED APERTURE
% ========================================================================
function [seg_id, centers, X, Y] = make_hex_segment_aperture(N, a, num_rings)

[xg, yg] = meshgrid(1:N,1:N);
cx = round((N+1)/2);
cy = round((N+1)/2);

X = xg - cx;
Y = cy - yg;

centers = [];
for q = -num_rings:num_rings
    for r = -num_rings:num_rings
        s = -q-r;
        if max(abs([q,r,s])) <= num_rings
            x0 = 1.5 * a * q;
            y0 = sqrt(3) * a * (r + q/2);
            centers(end+1,:) = [x0,y0]; %#ok<AGROW>
        end
    end
end

[~,ord] = sortrows([sqrt(sum(centers.^2,2)), -centers(:,2), centers(:,1)]);
centers = centers(ord,:);

seg_id = zeros(N,N);

for k = 1:size(centers,1)
    cxk = centers(k,1);
    cyk = centers(k,2);

    xv = cxk + a*cosd(0:60:300);
    yv = cyk + a*sind(0:60:300);

    colv = xv + cx;
    rowv = cy - yv;

    mk = poly2mask(colv, rowv, N, N);
    seg_id(mk) = k;
end

end


function E = make_random_segmented_field(seg_id, seg_centers, X, Y, seg_radius_pix, nSeg, zernike_nMax, coeff_sigma_rad)
%MAKE_RANDOM_SEGMENTED_FIELD Random segmented phase front from per-segment Zernikes.
%
% E = make_random_segmented_field(..., zernike_nMax, coeff_sigma_rad)
%
% zernike_nMax:
%   maximum radial Zernike order included on each segment.
%   0 = piston only
%   1 = piston + tip/tilt
%   2 = piston + tip/tilt + defocus/astigmatism
%   3 = + coma/trefoil
%   etc.
%
% coeff_sigma_rad:
%   scalar or vector of random coefficient standard deviations in radians.
%   If scalar, the same sigma is used for every included mode.
%   If vector, it must have one entry per generated real Zernike mode.
%
% Basis convention:
%   Real Zernikes on normalized local segment coordinates:
%       m = 0       : R_n^0(r)
%       m > 0, cos : R_n^m(r) cos(m theta)
%       m > 0, sin : R_n^m(r) sin(m theta)
%   Each mode is mean-subtracted except piston, then RMS-normalized on the
%   actual hexagonal segment pixels. Therefore coeff_sigma_rad is directly a
%   per-mode phase scale in radians over the segment mask.

if nargin < 7 || isempty(zernike_nMax)
    zernike_nMax = 2;
end
if nargin < 8 || isempty(coeff_sigma_rad)
    coeff_sigma_rad = 1.0;
end
if zernike_nMax < 0 || zernike_nMax ~= floor(zernike_nMax)
    error('zernike_nMax must be a nonnegative integer.');
end

modes = make_real_zernike_mode_list(zernike_nMax);
nModes = numel(modes);

if isscalar(coeff_sigma_rad)
    coeff_sigma_rad = coeff_sigma_rad * ones(nModes,1);
else
    coeff_sigma_rad = coeff_sigma_rad(:);
    if numel(coeff_sigma_rad) ~= nModes
        error('coeff_sigma_rad must be scalar or have one entry per mode. nModes = %d.', nModes);
    end
end

E = zeros(size(seg_id));

for k = 1:nSeg
    mk = seg_id == k;
    if ~any(mk(:)), continue; end

    cx = seg_centers(k,1);
    cy = seg_centers(k,2);

    dx = (X - cx) / seg_radius_pix;
    dy = (Y - cy) / seg_radius_pix;
    r = hypot(dx, dy);
    th = atan2(dy, dx);

    phi = zeros(size(seg_id));

    for j = 1:nModes
        n = modes(j).n;
        m = modes(j).m;
        kind = modes(j).kind;

        Z = real_zernike_nm(n, m, kind, r, th);

        if ~(n == 0 && m == 0)
            Z = Z - mean(Z(mk));
        end

        zrms = sqrt(mean(Z(mk).^2));
        if zrms <= eps || ~isfinite(zrms)
            continue;
        end
        Z = Z / zrms;

        coeff = coeff_sigma_rad(j) * randn();
        phi(mk) = phi(mk) + coeff * Z(mk);
    end

    E(mk) = exp(1i * phi(mk));
end
end

function modes = make_real_zernike_mode_list(nMax)
%MAKE_REAL_ZERNIKE_MODE_LIST Real Zernike modes through radial order nMax.
% Order is piston, then increasing n, then m. For m>0, cos then sin.

modes = struct('n', {}, 'm', {}, 'kind', {});
for n = 0:nMax
    for m = 0:n
        if mod(n - m, 2) ~= 0
            continue;
        end
        if m == 0
            modes(end+1) = struct('n', n, 'm', m, 'kind', 'm0'); %#ok<AGROW>
        else
            modes(end+1) = struct('n', n, 'm', m, 'kind', 'cos'); %#ok<AGROW>
            modes(end+1) = struct('n', n, 'm', m, 'kind', 'sin'); %#ok<AGROW>
        end
    end
end
end

function Z = real_zernike_nm(n, m, kind, r, th)
%REAL_ZERNIKE_NM Real-valued Zernike mode evaluated on r, theta.

R = zernike_radial(n, m, r);

switch lower(kind)
    case 'm0'
        Z = R;
    case 'cos'
        Z = R .* cos(m * th);
    case 'sin'
        Z = R .* sin(m * th);
    otherwise
        error('Unknown Zernike kind: %s', kind);
end
end

function R = zernike_radial(n, m, r)
%ZERNIKE_RADIAL Radial polynomial R_n^m(r), using the standard definition.

if m < 0 || m > n || mod(n - m, 2) ~= 0
    R = zeros(size(r));
    return;
end

R = zeros(size(r));
for s = 0:((n - m)/2)
    logc = gammaln(n - s + 1) ...
         - gammaln(s + 1) ...
         - gammaln((n + m)/2 - s + 1) ...
         - gammaln((n - m)/2 - s + 1);
    c = (-1)^s * exp(logc);
    R = R + c * r.^(n - 2*s);
end
end