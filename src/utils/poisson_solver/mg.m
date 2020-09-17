function [v] = mg(v, f, mask, vsz, opts)
%MG Multigrid for solving Poisson's equation with homogenous Dirichlet BCs.
%
%                           -Delta u = f, for x in mask
%                           u(~mask) = 0
%
%   [v] = MG(v, f, mask, vsz, [opts]);
%
%   Inputs
%   ------
%       v               initial guess (3d array).
%       f               right hand side (3d array): -Delta u = f.
%       mask            binary mask (3d array). 1 = interior, 0 = Dirichlet.
%       vsz             voxel size.
%
%       opts.tol        stopping tolerance for multigrid cycles.
%                       ||A*v_k - f||_2 <= tol * ||A*v_0 - f||_2
%                       default = sqrt(eps(class(f)))
%       opts.maxit      maximum number of multigrid cycles.
%                       default = max(size(f))
%       opts.mu         type of mu-cycle: 1 = V-cycle,  2 = W-cycle, ...
%                       default = 1
%       opts.npre       number of pre-relaxation sweeps.
%                       default = 1
%       opts.npost      number of post-relaxation sweeps.
%                       default = 1
%       opts.nboundary  number of extra boundary sweeps after interior sweeps
%                       on the downstroke and before interior sweeps on the
%                       upstroke.
%                       default = 2
%       opts.nlevels    number of multigrid levels.
%                       default = max(1, floor(log2(min(size(f)))) - 3)
%
%   Notes
%   -----
%       The Poisson equation is discretized using a standard second order
%       central finite-difference stencil.
%
%       The default smoother is a parallel red-black Gauss-Seidel. A serial
%       Gauss-Seidel is also implemented and can be used by setting the
%       gs_red_black flag in make.m to 0 and recompiling.
%
%       Coarse grids are generated such that a coarse grid point is a Dirichlet
%       point if any of its eight fine children is a Dirichlet point. The
%       coarse grid point is an interior point otherwise.
%
%       Prolongation and restriction operators are trilinear interpolation and
%       full-weighting (27 points), respectively. Prolongation and restriction
%       operators only prolong and restrict into interior grid points, a value
%       of 0 is assigned otherwise.
%
%       Inputs are automatically padded with a boundary layer of Dirichlet
%       points.
%
%   References
%   ----------
%       [1] Briggs, W. L., & McCormick, S. F. (2000). A multigrid tutorial
%       (Vol. 72). Siam.
%
%   See also FMG, MGPCG

    narginchk(4, 5);

    if nargin < 5, opts = []; end

    if ~isfield(opts, 'tol') || isempty(opts.tol)
        opts.tol = sqrt(eps(class(v)));
    end

    if ~isfield(opts, 'maxit') || isempty(opts.maxit)
        opts.maxit = max(size(v));
    end

    if ~isfield(opts, 'mu') || isempty(opts.mu)
        opts.mu = 1; % 1 = V-cycle, 2 = W-cycle
    end

    if ~isfield(opts, 'npre') || isempty(opts.npre)
        opts.npre = 1;
    end

    if ~isfield(opts, 'npost') || isempty(opts.npost)
        opts.npost = 1;
    end

    if ~isfield(opts, 'nboundary') || isempty(opts.nboundary)
        opts.nboundary = 2;
    end

    if ~isfield(opts, 'nlevels') || isempty(opts.nlevels)
        opts.nlevels = max(1, floor(log2(min(size(v)))) - 3);
    end


    validateinputs(v, f, mask, vsz, opts);

    if isa(v, 'single') || isa(f, 'single')
        v = cast(v, 'single');
        f = cast(f, 'single');
    end

    mask = logical(mask);
    vsz = double(vsz);


    v = mg_mex(v, f, mask, vsz, ...
               opts.tol, ...
               opts.maxit, ...
               opts.mu, ...
               opts.npre, ...
               opts.npost, ...
               opts.nboundary, ...
               opts.nlevels);

end


function [] = validateinputs(v, f, mask, vsz, o)

    sz = size(v);

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', 3, 'finite'};
    validateattributes(v, classes, attributes, mfilename, 'v', 1);

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', 3, 'size', sz, 'finite'};
    validateattributes(f, classes, attributes, mfilename, 'f', 2);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'size', sz, 'binary'};
    validateattributes(mask, classes, attributes, mfilename, 'mask', 3);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 4);

    % opts
    classes = {'numeric'};
    attributes = {'real', 'scalar', 'finite'};
    validateattributes(o.tol, classes, attributes, ...
        mfilename, 'opts.tol', 5);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0};
    validateattributes(o.maxit, classes, attributes, ...
        mfilename, 'opts.maxit', 5);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0};
    validateattributes(o.mu, classes, attributes, ...
        mfilename, 'opts.mu', 5);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>=', 0};
    validateattributes(o.npre, classes, attributes, ...
        mfilename, 'opts.npre', 5);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>=', 0};
    validateattributes(o.npost, classes, attributes, ...
        mfilename, 'opts.npost', 5);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>=', 0};
    validateattributes(o.nboundary, classes, attributes, ...
        mfilename, 'opts.nboundary', 5);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0, '<', floor(log2(min(sz)))-1};
    validateattributes(o.nlevels, classes, attributes, ...
        mfilename, 'opts.nlevels', 5);

end
