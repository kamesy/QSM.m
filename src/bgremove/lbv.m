function [fl] = lbv(f, mask, vsz, tol, maxit, mgopts, verbose)
%LBV Laplacian boundary value method.
%
%   [fl] = LBV(f, mask, vsz, [tol], [maxit], [mgopts], [verbose]);
%
%   Inputs
%   ------
%       f                   unwrapped field/phase (3d/4d array).
%       mask                binary mask of region of interest (3d array).
%       vsz                 voxel size.
%
%       tol                 stopping tolerance for conjugate gradient method.
%                           ||A*v_k - f||_2 <= tol * ||A*v_0 - f||_2
%                           default = 1e-8
%       maxit               maximum number of conjugate gradient iterations.
%                           default = ceil(sqrt(numel(mask)))
%       mgopts.maxit        maximum number of multigrid cycles.
%                           default = 2
%       mgopts.npre         number of pre-relaxation sweeps.
%                           default = 2
%       mgopts.npost        number of post-relaxation sweeps.
%                           default = 2
%       mgopts.nboundary    number of extra boundary sweeps after interior
%                           sweeps on the downstroke and before interior sweeps
%                           on the upstroke.
%                           default = 2
%       verbose             boolean flag for printing information for iterative
%                           solver.
%                           default: false
%
%   Outputs
%   -------
%       fl                  local field/phase.
%
%   Notes
%   -----
%       The Laplacian is computed using second order central finite
%       differences.
%
%       The resulting Poisson's equation is then solved inside the ROI with
%       homogenous Dirichlet BCs (f(~mask) = 0) [1] using a multigrid
%       preconditioned conjugate gradient method (mgpcg.m). The boundary of the
%       ROI is set such that values outside of it (mask = 0) are taken as
%       boundary points and values inside of it (mask = 1) as interior points.
%
%   References
%   ----------
%       [1] Zhou D, Liu T, Spincemaille P, Wang Y. Background field removal by
%       solving the Laplacian boundary value problem. NMR in Biomedicine. 2014
%       Mar;27(3):312-9.
%
%   See also MGPCG, UNWRAPLAPLACIAN, SHARP, VSHARP, RESHARP, ISMV

    narginchk(3, 7);

    if nargin < 7, verbose = false; end
    if nargin < 6 || isempty(mgopts), mgopts = []; end
    if nargin < 5 || isempty(maxit), maxit = ceil(sqrt(numel(mask))); end
    if nargin < 4 || isempty(tol), tol = 1e-8; end

    if ~isfield(mgopts, 'maxit') || isempty(mgopts.maxit)
        mgopts.maxit = 2;
    end

    if ~isfield(mgopts, 'npre') || isempty(mgopts.npre)
        mgopts.npre = 2;
    end

    if ~isfield(mgopts, 'npost') || isempty(mgopts.npost)
        mgopts.npost = 2;
    end

    if ~isfield(mgopts, 'nboundary') || isempty(mgopts.nboundary)
        mgopts.nboundary = 2;
    end


    validateinputs(f, mask, vsz, tol, maxit, mgopts, verbose);

    mask = logical(mask);
    vsz = double(vsz);

    % output
    fl = zeros(size(f), 'like', f);

    % crop to avoid unnecessary work. f(~mask) = 0
    [ix, iy, iz] = cropIndices(mask);

    % pad for finite diffy stencil
    ix = vec(max(1, ix(1)-1) : min(size(mask, 1), ix(end)+1));
    iy = vec(max(1, iy(1)-1) : min(size(mask, 2), iy(end)+1));
    iz = vec(max(1, iz(1)-1) : min(size(mask, 3), iz(end)+1));

    f = f(ix, iy, iz, :);
    mask = mask(ix, iy, iz);

    for t = 1:size(f, 4)
        fl(ix,iy,iz,t) = ...
            lbv_(f(:,:,:,t), mask, vsz, tol, maxit, mgopts, verbose);
    end

end


function [u] = lbv_(u, m, h, tol, maxit, mgopts, verbose)

    % compute Laplacian
    d2u = lap(u, m, h);

    % solve Poisson's equation with homogenous Dirichlet BCs:
    %   -Delta u = f,   for x in mask
    %   u(~mask) = 0
    u = mgpcg(-d2u, m, h, tol, maxit, mgopts, verbose);

end


function [] = validateinputs(f, mask, vsz, tol, maxit, o, verbose)

    sz = size(f);

    if ndims(f) < 4
        nd = 3;
    else
        nd = 4;
    end

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', nd, 'finite'};
    validateattributes(f, classes, attributes, mfilename, 'f', 1);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'size', sz(1:3), 'finite', 'binary'};
    validateattributes(mask, classes, attributes, mfilename, 'mask', 2);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 3);

    % cg opts
    classes = {'numeric'};
    attributes = {'real', 'scalar', 'finite'};
    validateattributes(tol, classes, attributes, mfilename, 'tol', 4);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0};
    validateattributes(maxit, classes, attributes, mfilename, 'maxit', 5);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'scalar', 'binary'};
    validateattributes(verbose, classes, attributes, mfilename, 'verbose', 7);

    % mg opts
    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0};
    validateattributes(o.maxit, classes, attributes, ...
        mfilename, 'opts.maxit', 6);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>=', 0};
    validateattributes(o.npre, classes, attributes, ...
        mfilename, 'opts.npre', 6);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>=', 0};
    validateattributes(o.npost, classes, attributes, ...
        mfilename, 'opts.npost', 6);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>=', 0};
    validateattributes(o.nboundary, classes, attributes, ...
        mfilename, 'opts.nboundary', 6);

end
