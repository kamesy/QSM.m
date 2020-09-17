function [fl, mask] = sharp(f, mask, vsz, r, thr)
%SHARP Sophisticated harmonic artifact reduction for phase data.
%
%   [fl, mask] = SHARP(f, mask, vsz, [r], [thr]);
%
%   Inputs
%   ------
%       f       unwrapped field/phase (3d/4d array).
%       mask    binary mask of region of interest (3d array).
%       vsz     voxel size for smv kernel.
%
%       r       radius of smv kernel in mm.
%               default: 9
%       thr     threshold for high pass filter.
%               default: 0.05
%
%   Outputs
%   -------
%       fl      local field/phase.
%       mask    eroded binary mask.
%
%   References
%   ----------
%       [1] Schweser F, Deistung A, Lehr BW, Reichenbach JR. Quantitative
%       imaging of intrinsic magnetic tissue properties using MRI signal phase:
%       an approach to in vivo brain iron metabolism?. Neuroimage. 2011 Feb
%       14;54(4):2789-807.
%
%   See also SMVKERNEL, VSHARP, RESHARP, ISMV, PDF, LBV

    narginchk(3, 5);

    if nargin < 5, thr = 0.05; end
    if nargin < 4 || isempty(r), r = 9; end


    % output
    fl = zeros(size(f), 'like', f);

    % crop image to mask and pad for convolution
    [ix, iy, iz] = cropIndices(mask);
    pad = ceil(r / min(vsz)) + [0, 0, 0];

    f = padfastfft(f(ix, iy, iz, :), pad);
    mask1 = padfastfft(mask(ix, iy, iz), pad);

    % get smv kernel
    h = smvKernel(size(mask1), r, vsz, class(f));
    h = real(fft3(ifftshift(h)));

    % erode mask
    mask1 = real(ifft3(h .* fft3(single(mask1)))) > 0.999;

    % sharp kernel
    h = 1 - h;

    % 4d support
    if ndims(f) > 3
        h = repmat(h, [1, 1, 1, size(f, 4)]);
        maskn = repmat(mask1, [1, 1, 1, size(f, 4)]);
    else
        maskn = mask1;
    end

    % SHARP
    f = maskn .* real(ifft3(h .* fft3(f)));
    f = fft3(f) ./ h;
    f(abs(h) < thr) = 0;
    f = maskn .* real(ifft3(f));

    % unpad and uncrop
    if ndims(f) > 3
        sz = [length(ix), length(iy), length(iz), size(f, 4)];
    else
        sz = [length(ix), length(iy), length(iz)];
    end

    fl(ix, iy, iz, :) = unpadfastfft(f, sz, pad);

    if nargout > 1
        mask(ix, iy, iz) = unpadfastfft(mask1, sz(1:3), pad);
    end

end
