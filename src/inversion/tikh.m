function [x] = tikh(f, mask, vsz, bdir, lambda)
%TIKH Tikhonov regularization
%
%   [x] = TIKH(f, mask, vsz, [bdir], [lambda]);
%
%   Inputs
%   ------
%       f       unwrapped local field/phase (3d/4d array).
%       mask    binary mask of region of interest (3d array).
%       vsz     voxel size.
%
%       bdir    unit vector of B field direction.
%               default: [0, 0, 1]
%       lambda  L*.L, where L is a tikhonov matrix.
%               default: 5e-3 .* psf2otf(-laplaceKernel(vsz), size(D))
%
%   Outputs
%   -------
%       x       susceptibility map.
%
%   References
%   ----------
%       [1] Bilgic B, Chatnuntawech I, Fan AP, Setsompop K, Cauley SF, Wald LL,
%       Adalsteinsson E. Fast image reconstruction with L2‐regularization.
%       Journal of magnetic resonance imaging. 2014 Jul;40(1):181-91.

    narginchk(3, 5);

    if nargin < 5, lambda = []; end
    if nargin < 4 || isempty(bdir), bdir = [0, 0, 1]; end


    % output
    x = zeros(size(f), 'like', f);

    D = dipoleKernel(size(mask), vsz, bdir, 'k', class(f));

    if isempty(lambda)
        lambda = -5e-3 .* psf2otf(laplaceKernel(vsz, class(f)), size(D));
    end

    ilhs = 1 ./ (conj(D) .* D + lambda);
    ilhs(~isfinite(ilhs)) = 0;

    f = fft3(f);

    % 4d multi-echo support
    for t = 1:size(f, 4)
        x(:,:,:,t) = mask .* real(ifft3(ilhs .* D .* f(:,:,:,t)));
    end

end
