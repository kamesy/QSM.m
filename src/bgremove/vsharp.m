function [fl, mask] = vsharp(f, mask, vsz, rs, thr)
%VSHARP Variable kernels SHARP.
%
%   [fl, mask] = VSHARP(f, mask, vsz, [rs], [thr]);
%
%   Inputs
%   ------
%       f       unwrapped field/phase (3d/4d array).
%       mask    binary mask of region of interest (3d array).
%       vsz     voxel size for smv kernel.
%
%       rs      radii of smv kernels in mm.
%               default: 9:-2*max(vsz):2*max(vsz)
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
%       [1] Wu B, Li W, Guidon A, Liu C. Whole brain susceptibility mapping
%       using compressed sensing. Magnetic resonance in medicine. 2012
%       Jan;67(1):137-47.
%
%   See also SMVKERNEL, SHARP, RESHARP, ISMV, PDF, LBV

    narginchk(3, 5);

    if nargin < 5, thr = 0.05; end
    if nargin < 4 || isempty(rs), rs = 9:-2*max(vsz):2*max(vsz); end


    %output
    fl = zeros(size(f), 'like', f);

    % crop image to mask and pad for convolution
    [ix, iy, iz] = cropIndices(mask);
    pad = ceil(max(vec(rs)) / min(vsz)) + [0, 0, 0];

    f = padfastfft(f(ix, iy, iz, :), pad);
    mask0 = padfastfft(mask(ix, iy, iz), pad);

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

    % deconvolution + high-pass filter
    f = maskn .* real(ifft3(ih .* fft3(f)));

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
