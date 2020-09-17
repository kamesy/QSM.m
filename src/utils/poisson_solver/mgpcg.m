function [v] = mgpcg(f, mask, vsz, tol, maxit, mgopts, verbose)
%MGPCG Multigrid preconditioned conjugate gradient for Poisson's equation
%   with homogenous Dirichlet BCs.
%
%                           -Delta u = f, for x in mask
%                           u(~mask) = 0
%
%   [v] = MGPCG(f, mask, vsz, [tol], [maxit], [mgopts], [verbose]);
%
%   Inputs
%   ------
%       f                   right hand side (3d array): -Delta u = f.
%       mask                binary mask (3d array). 1 = interior, 0 = Dirichlet.
%       vsz                 voxel size.
%
%       tol                 stopping tolerance for conjugate gradient method.
%                           ||A*v_k - f||_2 <= tol * ||A*v_0 - f||_2
%                           default = sqrt(eps(class(f)))
%       maxit               maximum number of conjugate gradient iterations.
%                           default = numel(f)
%       mgopts.tol          stopping tolerance for multigrid cycles.
%                           ||A*v_k - f||_2 <= tol * ||A*v_0 - f||_2
%                           default = -1 (fixed iterations)
%       mgopts.maxit        maximum number of multigrid cycles.
%                           default = 1
%       mgopts.mu           type of mu-cycle: 1 = V-cycle,  2 = W-cycle, ...
%                           default = 1
%       mgopts.npre         number of pre-relaxation sweeps.
%                           default = 1
%       mgopts.npost        number of post-relaxation sweeps.
%                           default = 1
%       mgopts.nboundary    number of extra boundary sweeps after interior
%                           sweeps on the downstroke and before interior sweeps
%                           on the upstroke.
%                           default = 2
%       mgopts.nlevels      number of multigrid levels.
%                           default = max(1, floor(log2(min(size(v)))) - 3)
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
%   See also CG, MG, FMG

    narginchk(3, 7);

    if nargin < 7, verbose = false; end
    if nargin < 6 || isempty(mgopts), mgopts = []; end
    if nargin < 5 || isempty(maxit), maxit = numel(f); end
    if nargin < 4 || isempty(tol), tol = sqrt(eps(class(f))); end

    if ~isfield(mgopts, 'tol') || isempty(mgopts.tol)
        mgopts.tol = -1; % fixed iterations
    end

    if ~isfield(mgopts, 'maxit') || isempty(mgopts.maxit)
        mgopts.maxit = 1;
    end

    if ~isfield(mgopts, 'mu') || isempty(mgopts.mu)
        mgopts.mu = 1; % 1 = V-cycle, 2 = W-cycle
    end

    if ~isfield(mgopts, 'npre') || isempty(mgopts.npre)
        mgopts.npre = 1;
    end

    if ~isfield(mgopts, 'npost') || isempty(mgopts.npost)
        mgopts.npost = 1;
    end

    if ~isfield(mgopts, 'nboundary') || isempty(mgopts.nboundary)
        mgopts.nboundary = 2;
    end

    if ~isfield(mgopts, 'nlevels') || isempty(mgopts.nlevels)
        mgopts.nlevels = max(1, floor(log2(min(size(f)))) - 3);
    end


    validateinputs(f, mask, vsz, tol, maxit, mgopts, verbose);

    mask = logical(mask);
    vsz = double(vsz);

    % initial guess
    v = zeros(size(f), 'like', f);

    v = mgpcg_mex(v, f, mask, vsz, ...
            tol, ...
            maxit, ...
            mgopts.tol, ...
            mgopts.maxit, ...
            mgopts.mu, ...
            mgopts.npre, ...
            mgopts.npost, ...
            mgopts.nboundary, ...
            mgopts.nlevels, ...
            verbose);

end


function [] = validateinputs(f, mask, vsz, tol, maxit, o, verbose)

    sz = size(f);

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', 3, 'size', sz, 'finite'};
    validateattributes(f, classes, attributes, mfilename, 'f', 1);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'size', sz, 'binary'};
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
    attributes = {'real', 'scalar', 'finite'};
    validateattributes(o.tol, classes, attributes, ...
        mfilename, 'opts.tol', 6);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0};
    validateattributes(o.maxit, classes, attributes, ...
        mfilename, 'opts.maxit', 6);

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0};
    validateattributes(o.mu, classes, attributes, ...
        mfilename, 'opts.mu', 6);

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

    classes = {'numeric'};
    attributes = {'scalar', 'integer', '>', 0, '<', floor(log2(min(sz)))-1};
    validateattributes(o.nlevels, classes, attributes, ...
        mfilename, 'opts.nlevels', 6);

end
