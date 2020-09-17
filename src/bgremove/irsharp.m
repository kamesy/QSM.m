function [fl, mask] = irsharp(f, fw, mask, vsz, rs, thr, slim)
%IRSHARP Improved region adaptive kernel SHARP
%
%   [fl, mask] = IRSHARP(f, fw, mask, vsz, [rs], [thr], [slim]);
%
%   Inputs
%   ------
%       f       unwrapped field/phase (3d/4d array).
%       fw      wrapped field/phase (3d/4d array).
%       mask    binary mask of region of interest (3d array).
%       vsz     voxel size for smv kernel.
%
%       rs      radii of smv kernels in mm.
%               default: 9:-2*max(vsz):2*max(vsz)
%       thr     threshold for high pass filter.
%               default: 0.05
%       slim    min, max sigma for spherical Gaussian weights.
%               default: [2, 8]
%
%   Outputs
%   -------
%       fl      local field/phase.
%       mask    eroded binary mask.
%
%   Notes
%   -----
%       DO NOT USE.
%
%       The method presented in [1] does not work. There are several
%       inconsistencies and the results presented cannot be reproduced using
%       the proposed algorithm.
%
%       The problem with the proposed method is the standard deviations of the
%       Gaussian kernels. The authors do not take into account the discrete
%       nature of the Gaussians. For example, sigma < 0.1 for a 1mm isotropic
%       grid is equivalent to convolving with a delta function and for the
%       smallest smv kernel, radius 1, sigma < 0.3 is effectively a delta
%       function. Furthermore, having an smv kernel with radius 5 and sigma 0.5
%       will give the same result as an smv kernel with radius 3 and both
%       results will be close to an smv kernel with radius 1 and sigma 0.5.
%
%       As the authors ignore this relationship, between kernel size and
%       standard deviation, as well as the rapid drop-off of the discrete
%       Gaussian kernels, the proposed method easily allows for convolutions
%       with delta functions (see Figure 6, C = 0.4).
%
%       On top of that, we are convolving with Gaussian kernels and then
%       deconvolving with normalized, constant, spherical kernels...
%
%   References
%   ----------
%       [1] Fang J, Bao L, Li X, van Zijl PC, Chen Z. Background field removal
%       for susceptibility mapping of human brain with large susceptibility
%       variations. Magnetic resonance in medicine. 2019 Mar;81(3):2025-37.
%
%   See also SMVKERNEL, SHARP, VSHARP, RESHARP, ISMV, PDF, LBV

    narginchk(4, 7);

    if nargin < 7, slim = [2, 8]; end
    if nargin < 6 || isempty(thr), thr = 0.05; end
    if nargin < 5 || isempty(rs), rs = 9:-2*max(vsz):2*max(vsz); end


    warning('DO NOT USE. read `help irsharp`');


    %output
    fl = zeros(size(f), 'like', f);

    % crop image to mask and pad for convolution
    [ix, iy, iz] = cropIndices(mask);
    pad = ceil(max(vec(rs)) / min(vsz)) + [0, 0, 0];

    f = padfastfft(f(ix, iy, iz, :), pad);
    fw = padfastfft(fw(ix, iy, iz, :), pad);
    mask0 = padfastfft(mask(ix, iy, iz), pad);

    % vsharp
    [f1, ih, maskn] = vsharp_(f, mask0, rs, thr, vsz);

    % correct each echo with irsharp
    for t = 1:size(f, 4)
        f1(:,:,:,t) = ...
            irsharp_(f1(:,:,:,t), f(:,:,:,t), fw(:,:,:,t), mask0, rs, slim, vsz);
    end

    % deconvolution + high-pass filter
    f = maskn .* real(ifft3(ih .* fft3(f1)));

    % unpad and uncrop
    if ndims(f) > 3
        sz = [length(ix), length(iy), length(iz), size(f, 4)];
    else
        sz = [length(ix), length(iy), length(iz)];
    end

    fl(ix, iy, iz, :) = unpadfastfft(f, sz, pad);

    if nargout > 1
        mask(ix, iy, iz) = unpadfastfft(maskn(:,:,:,1), sz(1:3), pad);
    end

end


function [f1] = irsharp_(f1, f, fw, m, rs, C, vsz)
% manual convolution with spherical Gaussian kernel. each voxel has a different
% kernel. we only convolve voxels above the threshold. others are already
% convolved with the smv kernel from vsharp.

    T = class(f1);
    sz = size(f1);
    m = logical(m);

    % get the weights
    sigma = sgkWeights(f, fw, m, vsz);

    % compute threshold. first paragraph pg 2029.
    % smax = max(vec(sigma));
    % smean = sum(vec(sigma)) / sum(vec(m));
    % s0 = smean + (smax - smean) / (smax + smean);

    % the original threshold is not robust to outliers and highly depends
    % on the + \epsilon = 1 in the sgkWeights, which it really shouldn't.
    % This should be more robust.
    s0 = prctile(sigma, 90);

    % find voxels to convolve
    I = find(sigma > s0);
    sigma = sigma(I);
    sigma = rescale(sigma, C(1), C(2));

    % prepare spherical kernel
    sk = {};
    for rr = 1:length(rs)
        r = rs(rr);
        sk{rr} = smvKernel(2*[r,r,r] + 1, r, vsz, T); %#ok<AGROW>
    end

    for ll = 1:length(I)
        [ii, jj, kk] = ind2sub(sz, I(ll));
        for rr = 1:length(rs)
            r = rs(rr);

            ix = ii-r : ii+r;
            iy = jj-r : jj+r;
            iz = kk-r : kk+r;

            if ix(1) < 1 || ix(end) > sz(1)
                continue;
            end

            if iy(1) < 1 || iy(end) > sz(2)
                continue;
            end

            if iz(1) < 1 || iz(end) > sz(3)
                continue;
            end

            % only convolve inside mask
            if all(vec(m(ix, iy, iz)))
                d = 2*[r,r,r] + 1;

                sgk = gaussianKernel(d, sigma(ll), vsz, [], T);
                sgk = sk{rr} .* sgk;

                n = sum(vec(sgk));
                if n ~= 0
                    sgk = sgk / n;
                end

                sgk = sgk .* f(ix, iy, iz);
                f1(ii, jj, kk) = f(ii, jj, kk) - sum(vec(sgk));

                break;
            end
        end
    end

end


function [sigma] = sgkWeights(f, fw, m, vsz)
% Spherical Gaussian Kernel. pg. 2028
%   The paper does not specify how to `combine` the three components of the
%   gradient. Here, the 2-norm is used.
%   K is normalized from 0 to 1 in the paper but the other two terms
%   are not. Here, all terms are normalized.
%   The + \epsilon term, which is set to 1 in the paper, is removed.

    [dx, dy, dz] = gradc(f, m, vsz);        % 1 ./ Equation (6)
    G = sqrt(dx.^2 + dy.^2 + dz.^2);        % 2-norm
    G = G ./ max(vec(G));                   % normalizing
    G(m & G == 0) = 1;                      % avoid div by 0

    K = m .* abs((f - fw)./(2*pi));         % 1 ./ Equation (8)
    K = K ./ max(vec(K));                   % normalizing
    K(m & K == 0) = 1;                      % avoid div by 0

    f = m .* abs(f);                        % 1 ./ Equation (7)
    f = f ./ max(vec(f));                   % normalizing
    f(m & f == 0) = 1;                      % avoid div by 0

    sigma = G .* f .* K;                    % 1 ./ Equation (9)

end


function [f, ih, maskn] = vsharp_(f, mask0, rs, thr, vsz)
% standard vsharp; copied to return the deconvolution kernel and masks
%   See also VSHARP

    % pre-compute ffts
    F = fft3(f);
    M = fft3(single(mask0));

    % temporary variables
    f = zeros(size(f), 'like', f);
    mask0 = logical(f);

    % sort radii, largest first
    rs = sort(vec(rs), 1, 'descend');

    % using conjugate symmetry properties of fourier transform to fft two
    % kernels at once:
    %   x(t) even, real => X(w) real, even
    %   x(t) even, imag => X(w) imag, even
    % So,
    %       real(fft(s1 + i*s2)) = fft(s1)
    %       imag(fft(s1 + i*s2)) = fft(s2)
    for ii = 1:2:length(rs)

        % do two kernels at once
        if ii+1 <= length(rs)

            % get first smv kernel
            r1 = rs(ii);
            h = smvKernel(size(M), r1, vsz, class(f));

            % get second smv kernel
            r2 = rs(ii+1);
            h = h + 1i*smvKernel(size(M), r2, vsz, class(f));

            h = fft3(ifftshift(h));

            % erode both masks
            m = ifft3(h .* M);
            mask1 = real(m) > 0.999;
            mask2 = imag(m) > 0.999;

            % sharp kernel
            h = (1 + 1i) - h;

            % high-pass filter first (largest) kernel for deconvolution
            if ii == 1
                ih = 1 ./ real(h);
                ih(abs(real(h)) < thr) = 0;
                if ndims(f) > 3, ih = repmat(ih, [1, 1, 1, size(f, 4)]); end
            end

            % 4d support
            if ndims(f) > 3
                h = repmat(h, [1, 1, 1, size(f, 4)]);
                mask1n = repmat(mask1, [1, 1, 1, size(f, 4)]);
                mask2n = repmat(mask2, [1, 1, 1, size(f, 4)]);
            else
                mask1n = mask1;
                mask2n = mask2;
            end

            % SHARP
            hf = ifft3(h .* F);

            f = f + (mask1n - mask0) .* real(hf) + (mask2n - mask1n) .* imag(hf);
            mask0 = mask2n;
            maskn = mask2n;

        % do single kernel
        else
            % get smv kernel
            r = rs(ii);
            h = smvKernel(size(M), r, vsz, class(f));
            h = real(fft3(ifftshift(h)));

            % erode mask
            mask1 = real(ifft3(h .* M)) > 0.999;

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
            f = f + (maskn - mask0) .* real(ifft3(h .* F));
        end
    end

end
