function [v] = fmg(f, mask, vsz, opts)
%FMG Full multigrid for solving Poisson's equation with homogenous Dirichlet BCs.
%
%                           -Delta u = f, for x in mask
%                           u(~mask) = 0
%
%   [v] = FMG(f, mask, vsz, [opts]);
%
%   Inputs
%   ------
%       f               right hand side (3d array): -Delta u = f.
%       mask            binary mask (3d array). 1 = interior, 0 = Dirichlet.
%       vsz             voxel size.
%
%       opts.tol        stopping tolerance for multigrid cycles on the finest
%                       level. Coarse levels do not have a stopping tolerance.
%                       ||A*v_k - f||_2 <= tol * ||A*v_0 - f||_2
%                       default = sqrt(eps(class(f)))
%       opts.maxit      maximum number of multigrid cycles for coarse levels
%                       and for the finest level if opts.tol <= 0. If
%                       opts.tol > 0 then on the finest level the maximum
%                       number of iterations is 100*opts.maxit. This should
%                       be an input parameter but is not yet implemented.
%                       default = 1
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
%   See also MG, MGPCG

    narginchk(3, 4);

    if nargin < 4, opts = []; end

    if ~isfield(opts, 'tol') || isempty(opts.tol)
        opts.tol = sqrt(eps(class(f)));
    end

    if ~isfield(opts, 'maxit') || isempty(opts.maxit)
        opts.maxit = 1;
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
        opts.nlevels = max(1, floor(log2(min(size(f)))) - 3);
    end


    validateinputs(f, mask, vsz, opts);

    mask = logical(mask);
    vsz = double(vsz);

    v = fmg_mex(f, mask, vsz, ...
                opts.tol, ...
                opts.maxit, ...
                opts.mu, ...
                opts.npre, ...
                opts.npost, ...
                opts.nboundary, ...
                opts.nlevels);

end


function [] = validateinputs(f, mask, vsz, o)

    sz = size(f);

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', 3, 'finite'};
    validateattributes(f, classes, attributes, mfilename, 'v', 1);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'size', sz, 'binary'};
    validateattributes(mask, classes, attributes, mfilename, 'mask', 2);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 3);

    % opts
    classes = {'numeric'};
    attributes = {'real', 'scalar', 'finite'};
    validateattributes(o.tol, classes, attributes, ...
        mfilename, 'opts.tol', 4);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0};
    validateattributes(o.maxit, classes, attributes, ...
        mfilename, 'opts.maxit', 4);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0};
    validateattributes(o.mu, classes, attributes, ...
        mfilename, 'opts.mu', 4);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>=', 0};
    validateattributes(o.npre, classes, attributes, ...
        mfilename, 'opts.npre', 4);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>=', 0};
    validateattributes(o.npost, classes, attributes, ...
        mfilename, 'opts.npost', 4);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>=', 0};
    validateattributes(o.nboundary, classes, attributes, ...
        mfilename, 'opts.nboundary', 4);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0, '<', floor(log2(min(sz)))-1};
    validateattributes(o.nlevels, classes, attributes, ...
        mfilename, 'opts.nlevels', 4);

end
