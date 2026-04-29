%% demo_circles.m
% Demo for Raz-style recovery of Im(A1^*A2) from Re(A1^*A2)
% using a geometry that preserves orderedness/causality on every x-line.
%
% Geometry:
%   E1 = large circular aperture centered at (0,0)
%   E2 = small circular aperture placed to the RIGHT of E1, tangent externally
%
% This means that on every y-line where both are present:
%   support(E1) is a left interval
%   support(E2) is a right interval
% so orderedness is preserved.
%
% NOTE:
%   This uses the function:
%       raz_im_from_re_general(ReC, supp1, supp2)
%   from the previous message.

clear; close all; clc;

%% Grid
N = 512;
[x, y] = meshgrid(linspace(-1,1,N), linspace(-1,1,N));

%% Aperture geometry
R1 = 0.35;      % large radius
R2 = 0.08;      % small radius

% Large circle center
x1 = 0.0;
y1 = 0.0;

% Small circle center:
% To preserve causality on x-lines, put it externally tangent on +x side.
% (If you place the small-circle center on the large-circle circumference,
% the supports generally overlap and orderedness is lost.)
x2 = x1 + R1 + 0*R2;
y2 = 0.0;

supp_large = ((x - x1).^2 + (y - y1).^2) <= R1^2;
supp_small = ((x - x2).^2 + (y - y2).^2) <= R2^2;

supp1 = supp_large & ~supp_small;
supp2 = supp_large &  supp_small;

if any(supp1(:) & supp2(:))
    error('Supports overlap. Geometry is not causal.');
end

%% Fields
% Keep this simple: unit amplitude with smooth phase on each support
rng(1);

sigma_filter_px = 4;

phi = 5 * imgaussfilt(randn(N), sigma_filter_px);
amp = ones(N);   % or smooth amplitude if you want

E = supp_large .* amp .* exp(1i * phi);

E1 = E .* supp1;
E2 = E .* supp2;

%% Fourier transforms
A1 = fft2(E1);
A2 = fft2(E2);

C_true  = A1 .* conj(A2);
ReC     = real(C_true);
ImC_true = imag(C_true);

%% Recover Im(C) from Re(C)
[ImC_est, C_est, info] = raz_im_from_re_general(ReC, supp1.', supp2.');
% transpose supports because the solver expects supports as [Nx, Ny]
% with x along rows / first dimension. Here x is columns in imagesc convention.

valid_cols = find(info.valid_line);
fprintf('Valid lines: %d / %d\n', numel(valid_cols), numel(info.valid_line));

if isempty(valid_cols)
    error('No valid causal lines found.');
end

%% Error metrics on valid lines only
mask_valid = false(size(ImC_true));
mask_valid(:, valid_cols) = true;

rel_err = norm(ImC_est(mask_valid) - ImC_true(mask_valid)) / ...
          max(norm(ImC_true(mask_valid)), eps);

fprintf('Relative error on valid lines: %.3e\n', rel_err);

%% Display supports
figure('Color','w','Position',[100 100 1200 700]);

subplot(2,3,1);
imagesc(supp1 + 2*supp2);
axis image; colorbar;
title('Supports: E1 (1), E2 (2)');

subplot(2,3,2);
imagesc(angle(E1 + E2));
axis image; colorbar;
title('Phase of E1+E2');

subplot(2,3,3);
plot(info.valid_line, '.-');
ylim([-0.1 1.1]); grid on;
title('Valid causal lines');
xlabel('line index'); ylabel('valid');

subplot(2,3,4);
imagesc(log10(abs(ReC) + 1e-9));
axis image; colorbar;
title('log_{10}|Re(C)|');

subplot(2,3,5);
imagesc(log10(abs(ImC_true) + 1e-9));
axis image; colorbar;
title('log_{10}|Im(C) true|');

subplot(2,3,6);
imagesc(log10(abs(ImC_est) + 1e-9));
axis image; colorbar;
title('log_{10}|Im(C) recovered|');

sgtitle('Raz causal-circle demo');

%% Compare one valid line
k = valid_cols(round(numel(valid_cols)/2));

figure('Color','w');
plot(ImC_true(:,k), 'k-', 'LineWidth', 1.5); hold on;
plot(ImC_est(:,k),  'r--', 'LineWidth', 1.5);
grid on;
legend('True','Recovered');
title(sprintf('Imaginary part on valid line %d', k));
xlabel('x-frequency index');
ylabel('Im(C)');

%% Optional: inspect inverse FFT support on one valid line
q_true = ifft(C_true(:,k));
q_est  = ifft(C_est(:,k));

figure('Color','w');
plot(abs(q_true), 'k-', 'LineWidth', 1.5); hold on;
plot(abs(q_est),  'r--', 'LineWidth', 1.5);
grid on;
legend('|ifft(C_{true})|','|ifft(C_{est})|');
title(sprintf('Inverse-FFT support on line %d', k));
xlabel('index');
ylabel('magnitude');