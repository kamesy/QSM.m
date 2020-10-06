function [fl] = pdf(f, mask, vsz, w, bdir, lambda, tol, maxit, verbose)
%PDF Projection onto dipole fields.
%
%   [fl] = PDF(f, mask, vsz, [w], [bdir], [lambda], [tol], [maxit], [verbose]);
%
%   Inputs
%   ------
%       f           unwrapped field/phase (3d/4d array).
%       mask        binary mask of region of interest (3d array).
%       vsz         voxel size for dipole kernel.
%
%       w           image space weights.
%                   default: mask.
%       bdir        unit vector of B field direction.
%                   default: [0, 0, 1]
%       lambda      regularization parameter.
%                   default: 0
%       tol         stopping tolerance for iterative solver.
%                   default: 1e-5
%       maxit       maximum number of iterations for iterative solver.
%                   default: ceil(sqrt(numel(mask)))
%       verbose     boolean flag for printing information for iterative solver.
%                   default: false
%
%   Outputs
%   -------
%       fl          local field/phase.
%
%   References
%   ----------
%       [1] Liu T, Khalidov I, de Rochefort L, Spincemaille P, Liu J,
%       Tsiouris AJ, Wang Y. A novel background field removal method for MRI
%       using projection onto dipole fields. NMR in Biomedicine. 2011
%       Nov;24(9):1129-36.
%
%   See also LSMR, SHARP, VSHARP, RESHARP, ISMV, LBV

    narginchk(3, 9);

    if nargin < 9, verbose = false; end
    if nargin < 8 || isempty(maxit), maxit = ceil(sqrt(numel(mask))); end
    if nargin < 7 || isempty(tol), tol = 1e-5; end
    if nargin < 6 || isempty(lambda), lambda = 0; end
    if nargin < 5 || isempty(bdir), bdir = [0, 0, 1]; end
    if nargin < 4 || isempty(w), w = []; end


    if isempty(w) || ~any(vec(w))
        w = mask;
    end

    % output
    fl = zeros(size(f), 'like', f);

    % get dipole kernel
    D = dipoleKernel(size(mask), vsz, bdir, 'i', class(f));
    D = psf2otf(D);

    % weights
    m = mask .* w;

    % operator
    function [v] = A(v, t)
        v = reshape(v, size(D));
        if t(1) == 1 || t(1) == 'n'
            v = m .* real(ifft3(D .* fft3(~m .* v)));
        else
            v = ~m .* real(ifft3(conj(D) .* fft3(m .* v)));
        end
        v = vec(v);
    end

    % loop over echoes
    for t = 1:size(f, 4)
        % rhs
        b = m .* f(:,:,:,t);

        % PDF
        xb = lsmr(@A, vec(b), lambda, tol, tol, [], maxit, [], verbose);
        xb = reshape(xb, size(D));

        fl(:,:,:,t) = ...
            logical(m) .* (f(:,:,:,t) - real(ifft3(D .* fft3(~m .* xb))));
    end

end
