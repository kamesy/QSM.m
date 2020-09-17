function [uphas, d2phas] = unwrapLaplacian(phas, mask, vsz)
%UNWRAPLAPLACIAN Laplacian phase unwrapping.
%
%   [uphas, d2phas] = UNWRAPLAPLACIAN(phas, mask, vsz);
%
%   Inputs
%   ------
%       phas    wrapped phase (3d/4d array).
%       mask    binary mask of region of interest (3d array).
%       vsz     voxel size.
%
%   Outputs
%   -------
%       uphas   unwrapped local phase.
%       d2phas  Laplacian of unwrapped phase.
%
%   Notes
%   -----
%       The Laplacian is computed using second order central finite differences
%       on the complex phase (see lapw.m).
%
%       The resulting Poisson's equation is then solved inside the ROI with
%       homogenous Dirichlet BCs (uphas(~mask) = 0) [2] using a multigrid
%       preconditioned conjugate gradient method (mgpcg.m). The boundary of the
%       ROI is set such that values outside of it (mask = 0) are taken as
%       boundary points and values inside of it (mask = 1) as interior points.
%
%       This method combines phase unwrapping [1] and harmonic background field
%       removing [2] (see lbv.m).
%
%   References
%   ----------
%       [1] Schofield MA, Zhu Y. Fast phase unwrapping algorithm for
%       interferometric applications. Optics letters. 2003 Jul 15;28(14):1194-6.
%
%       [2] Zhou D, Liu T, Spincemaille P, Wang Y. Background field removal by
%       solving the Laplacian boundary value problem. NMR in Biomedicine. 2014
%       Mar;27(3):312-9.
%
%   See also LAPW, MGPCG

    narginchk(3, 3);

    validateinputs(phas, mask, vsz);

    mask = logical(mask);
    vsz = double(vsz);


    % output
    uphas = zeros(size(phas), 'like', phas);

    if nargout > 1
        d2phas = zeros(size(phas), 'like', phas);
    end

    % crop to avoid unnecessary work. phas(~mask) = 0
    [ix, iy, iz] = cropIndices(mask);

    % pad for finite diffy stencil
    ix = vec(max(1, ix(1)-1) : min(size(mask, 1), ix(end)+1));
    iy = vec(max(1, iy(1)-1) : min(size(mask, 2), iy(end)+1));
    iz = vec(max(1, iz(1)-1) : min(size(mask, 3), iz(end)+1));

    mask = mask(ix, iy, iz);

    for t = 1:size(phas, 4)
        [uphas(ix,iy,iz,t), d2p] = unwrap_(phas(ix,iy,iz,t), mask, vsz);

        if nargout > 1
            d2phas(ix,iy,iz,t) = d2p;
        end
    end

end


function [u, d2u] = unwrap_(u, m, h)

    % get Laplacian of unwrapped phase from wrapped phase
    d2u = lapw(u, h);

    % solve Poisson's equation with homogenous Dirichlet BCs:
    %   -Delta u = f,   for x in mask
    %   u(~mask) = 0

    % TODO: input params
    tolcg = 1e-8;
    maxitcg = ceil(sqrt(numel(m)));

    mgopts.maxit = 2;
    mgopts.npre = 2;
    mgopts.npost = 2;
    mgopts.nboundary = 2;

    u = mgpcg(-m.*d2u, m, h, tolcg, maxitcg, mgopts);

end


function [] = validateinputs(phas, mask, vsz)

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

end
