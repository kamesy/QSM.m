function [fl, mask] = ismv(f, mask, vsz, r, tol, maxit, verbose)
%ISMV Iterative spherical mean value method.
%
%   [fl, mask] = ISMV(f, mask, vsz, [r], [tol], [maxit], [verbose]);
%
%   Inputs
%   ------
%       f           unwrapped field/phase (3d/4d array).
%       mask        binary mask of region of interest (3d array).
%       vsz         voxel size for smv kernel.
%
%       r           radius of smv kernel in mm.
%                   default: 3
%       tol         stopping tolerance.
%                   default: sqrt(eps(class(f)))
%       maxit       maximum number of iterations.
%                   default: 500
%       verbose     boolean flag for printing information each iteration.
%                   default: false
%
%   Outputs
%   -------
%       fl          local field/phase.
%       mask        eroded binary mask.
%
%   References
%   ----------
%       [1] Wen Y, Zhou D, Liu T, Spincemaille P, Wang Y. An iterative
%       spherical mean value method for background field removal in MRI.
%       Magnetic resonance in medicine. 2014 Oct;72(4):1065-71.
%
%   See also CG, SMVKERNEL, SHARP, VSHARP, RESHARP, PDF, LBV

    narginchk(3, 7);

    if nargin < 7, verbose = false; end
    if nargin < 6 || isempty(maxit), maxit = 500; end
    if nargin < 5 || isempty(tol), tol = sqrt(eps(class(f))); end
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
    m = real(ifft3(h .* fft3(single(mask1)))) > 0.999;

    % loop over echoes
    for t = 1:size(f, 4)
        if verbose && size(f, 4) > 1
            fprintf('Echo: %d/%d\n', t, size(f, 4));
        end

        ft = f(:,:,:,t);
        bc = (mask1 - m) .* ft;

        fb = ismv_(ft, h, bc, m, tol, maxit, verbose);
        fl_ = m .* (ft - fb);

        % unpad and uncrop
        fl(ix, iy, iz, t) = unpadfastfft(fl_, sz, pad);
    end

    if nargout > 1
        mask(ix, iy, iz) = unpadfastfft(m, sz(1:3), pad);
    end

end


function [f] = ismv_(f, h, bc, m, tol, maxit, verbose)

    res = norm2(m .* f);
    reltol = res * tol;

    if verbose
        fprintf('iter%6s\tresidual\n', '');
    end

    for ii = 1:maxit
        if res <= reltol
            break;
        end

        f0 = f;

        f = bc + m .* real(ifft3(h .* fft3(f)));

        res = norm2(f0 - f);

        if verbose
            fprintf('%5d/%d\t%1.3e\n', ii, maxit, res);
        end
    end

    if verbose
        fprintf('\n');
    end

end
