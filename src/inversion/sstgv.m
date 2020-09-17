function [x] = sstgv(phas, mask, vsz, lambda, alpha, tol, maxit, verbose)
%SSTGV Single-step total generalized variation.
%
%   [x] = SSTGV(phas, mask, vsz, [lambda], [alpha], [tol], [maxit], [verbose]);
%
%   Inputs
%   ------
%       phas        wrapped phase (3d/4d array).
%       mask        binary mask of region of interest (3d array).
%       vsz         voxel size.
%
%       lambda      regularization parameter.
%                   alpha1 = lambda, alpha0 = alpha*lambda
%                   default: 1e-4
%       alpha       ratio of TGV terms. if alpha < eps uses total variation
%                   instead of total generalized variation.
%                   alpha1 = lambda, alpha0 = alpha*lambda
%                   default: 3
%       tol         stopping tolerance.
%                   ||x_{k+1} - x_{k}||_2 <= tol * ||x_{k+1}||_2
%                   default = 3e-4
%       maxit       maximum number of iterations.
%                   default = 5000
%       verbose     flag for printing information for primal dual
%                   method. if verbose is greater than zero, print frequency is
%                   given by `maxit / min(max(verbose, 4), maxit)`
%                   default = 1
%
%   Outputs
%   -------
%       x           susceptibility map.
%
%   References
%   ----------
%       [1] Langkammer C, Bredies K, Poser BA, Barth M, Reishofer G, Fan AP,
%       Bilgic B, Fazekas F, Mainero C, Ropele S. Fast quantitative
%       susceptibility mapping using 3D EPI and total generalized variation.
%       Neuroimage. 2015 May 1;111:622-30.

    narginchk(3, 8);

    if nargin < 8, verbose = true; end
    if nargin < 7 || isempty(maxit), maxit = 5000; end
    if nargin < 6 || isempty(tol), tol = 3e-4; end
    if nargin < 5 || isempty(alpha), alpha = 3; end
    if nargin < 4 || isempty(lambda), lambda = 1e-4; end


    validateinputs(phas, mask, vsz, lambda, alpha, tol, maxit, verbose);

    mask = logical(mask);
    vsz = double(vsz);

    % output
    x = zeros(size(phas), 'like', phas);

    % crop to avoid unnecessary work. x(~mask) = 0
    [ix, iy, iz] = cropIndices(mask);

    % pad for finite diffy stencil
    ix = vec(max(1, ix(1)-2) : min(size(mask, 1), ix(end)+2));
    iy = vec(max(1, iy(1)-2) : min(size(mask, 2), iy(end)+2));
    iz = vec(max(1, iz(1)-2) : min(size(mask, 3), iz(end)+2));

    phas = phas(ix, iy, iz, :);
    mask = mask(ix, iy, iz);

    for t = 1:size(phas, 4)
        x(ix,iy,iz,t) = ...
            sstgv_(phas(:,:,:,t), mask, vsz, lambda, alpha, tol, maxit, verbose);
    end

end


function [x, d2u] = sstgv_(u, m, h, lam, alpha, tol, maxit, verbose)

    % compute Laplacian
    d2u = lapw(u, h);

    if alpha > eps
        % tgv minimization
        x = pd_tgv_mex(-d2u, m, h, lam, alpha, tol, maxit, verbose);
    else
        % tv minimization
        x = pd_tv_mex(-d2u, m, h, lam, tol, maxit, verbose);
    end

end


function [] = validateinputs(phas, mask, vsz, lambda, alpha, tol, maxit, verbose)

    sz = size(phas);

    if ndims(phas) < 4
        nd = 3;
    else
        nd = 4;
    end

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', nd, 'finite'};
    validateattributes(phas, classes, attributes, mfilename, 'phas', 1);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'size', sz(1:3), 'finite', 'binary'};
    validateattributes(mask, classes, attributes, mfilename, 'mask', 2);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 3);

    classes = {'numeric'};
    attributes = {'real', 'scalar', 'finite', '>', 0};
    validateattributes(lambda, classes, attributes, mfilename, 'lambda', 4);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'scalar', 'finite'};
    validateattributes(alpha, classes, attributes, mfilename, 'alpha', 5);

    classes = {'numeric'};
    attributes = {'real', 'scalar', 'finite'};
    validateattributes(tol, classes, attributes, mfilename, 'tol', 6);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0};
    validateattributes(maxit, classes, attributes, mfilename, 'maxit', 7);

    if ~islogical(verbose)
        classes = {'numeric'};
        attributes = {'scalar', 'integer', '>=', 0, '<=', maxit};
        validateattributes(verbose, classes, attributes, mfilename, 'verbose', 8);
    end

end
