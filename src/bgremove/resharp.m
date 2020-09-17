function [fl, mask] = resharp(f, mask, vsz, r, lambda, tol, maxit)
%RESHARP Regularization enabled SHARP.
%
%   [fl, mask] = RESHARP(f, mask, vsz, [r], [lambda], [tol], [maxit]);
%
%   Inputs
%   ------
%       f           unwrapped field/phase (3d/4d array).
%       mask        binary mask of region of interest (3d array).
%       vsz         voxel size for smv kernel.
%
%       r           radius of smv kernel in mm.
%                   default: 3
%       lambda      regularization parameter.
%                   default: 1e-2
%       tol         stopping tolerance for iterative solver.
%                   default: sqrt(eps(class(f)))
%       maxit       maximum number of iterations for iterative solver.
%                   default: ceil(sqrt(numel(mask)))
%
%   Outputs
%   -------
%       fl          local field/phase.
%       mask        eroded binary mask.
%
%   References
%   ----------
%       [1] Sun H, Wilman AH. Background field removal using spherical mean
%       value filtering and Tikhonov regularization. Magnetic resonance in
%       medicine. 2014 Mar;71(3):1151-7.
%
%   See also CGS, SMVKERNEL, SHARP, VSHARP, ISMV, PDF, LBV

    narginchk(3, 7);

    if nargin < 7, maxit = ceil(sqrt(numel(mask))); end
    if nargin < 6 || isempty(tol), tol = sqrt(eps(class(f))); end
    if nargin < 5 || isempty(lambda), lambda = 1e-2; end
    if nargin < 4 || isempty(r), r = 3; end


    % output
    fl = zeros(size(f), 'like', f);

    % crop image to mask and pad for convolution
    [ix, iy, iz] = cropIndices(mask);
    sz = [length(ix), length(iy), length(iz)];
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

    % operator
    function [v] = A(v)
        v = reshape(v, size(h));
        v = mask1 .* real(ifft3(h .* fft3(v))) + lambda.*v;
        v = vec(v);
    end

    % loop over echoes
    for t = 1:size(f, 4)
        % rhs
        b = mask1 .* real(ifft3(h .* fft3(f(:,:,:,t))));

        % RESHARP
        fl_ = cgs(@A, vec(b), tol, maxit);
        fl_ = reshape(fl_, size(h));
        fl_ = mask1 .* fl_;

        % unpad and uncrop
        fl(ix, iy, iz, t) = unpadfastfft(fl_, sz, pad);
    end

    if nargout > 1
        mask(ix, iy, iz) = unpadfastfft(mask1, sz, pad);
    end

end
