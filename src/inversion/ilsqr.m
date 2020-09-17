function [x, xsa, xfs, xlsqr] = ilsqr(f, mask, vsz, bdir, tol, maxit, verbose)
%ILSQR A method for estimating and removing streaking artifacts.
%
%   [x, xsa, xfs, xlsqr] =
%       ILSQR(f, mask, vsz, [bdir], [tol], [maxit], [verbose]);
%
%   Inputs
%   ------
%       f           unwrapped local field/phase (3d/4d array).
%       mask        binary mask of region of interest (3d array).
%       vsz         voxel size.
%
%       bdir        unit vector of B field direction.
%                   default: [0, 0, 1]
%       tol         stopping tolerance for final lsqr solver (xsa).
%                   default: 1e-2
%       maxit       maximum number of iterations for final lsqr (xsa).
%                   default: 50
%       verbose     boolean flag for printing information each iteration.
%                   default: true
%
%   Outputs
%   -------
%       x           susceptibility map.
%       xsa         estimated streaking artifacts.
%       xfs         fast qsm susceptibility map.
%       xlsqr       initial lsqr susceptibility map.
%
%   References
%   ----------
%       [1] Li W, Wang N, Yu F, Han H, Cao W, Romero R, Tantiwongkosi B, Duong
%       TQ, Liu C. A method for estimating and removing streaking artifacts in
%       quantitative susceptibility mapping. Neuroimage. 2015 Mar 1;108:111-22.

    narginchk(3, 7);

    if nargin < 7, verbose = true; end
    if nargin < 6 || isempty(maxit), maxit = 50; end
    if nargin < 5 || isempty(tol), tol = 1e-2; end
    if nargin < 4 || isempty(bdir), bdir = [0, 0, 1]; end


    % output
    x = zeros(size(f), 'like', f);
    if nargout > 1, xsa = zeros(size(f), 'like', f); end
    if nargout > 2, xfs = zeros(size(f), 'like', f); end
    if nargout > 3, xlsqr = zeros(size(f), 'like', f); end

    D = dipoleKernel(size(mask), vsz, bdir, 'k', class(f));

    % 4d multi-echo loop
    for t = 1:size(f, 4)
        if verbose && size(f, 4) > 1
            fprintf('Echo: %d/%d\n', t, size(f, 4));
        end

        [x(:,:,:,t), xsa_, xfs_, xlsqr_] = ...
            ilsqr_(f(:,:,:,t), mask, D, vsz, tol, maxit, verbose);

        if nargout > 1, xsa(:,:,:,t) = xsa_; end
        if nargout > 2, xfs(:,:,:,t) = xfs_; end
        if nargout > 3, xlsqr(:,:,:,t) = xlsqr_; end
    end

end


function [x, xsa, xfs, xlsqr] = ilsqr_(f, m, D, vsz, tol, maxit, verbose)

    % step 1: lsqr
    xlsqr = lsqr_(f, m, vsz, D);

    % step 2: fastqsm
    xfs = fastqsm_(f, m, vsz, D);

    % step 3: susceptibility artifacts
    xsa = susceptibilityArtifacts_(xlsqr, xfs, m, vsz, D, tol, maxit, verbose);

    % step 4: subtract artifacts
    x = m .* (xlsqr - xsa);

end


function [x] = lsqr_(f, mask, vsz, D)

    % TODO: input args
    % percentiles for laplacian weights
    pmin = 60;
    pmax = 99.9;

    tollsqr = 0.01;
    maxitlsqr = 50;

    sz = size(f);

    % Equation (7)
    w = laplacianWeightsIlsqr(f, mask, vsz, pmin, pmax);

    % Equation (6)
    b = D .* fft3(w .* f);

    [x, ~] = lsqr(@afun, vec(b), tollsqr, maxitlsqr);

    x = reshape(x, sz);
    x = mask .* real(ifft3(x));

    function [x] = afun(x, ~)
        x = reshape(x, sz);
        x = D .* fft3(w .* real(ifft3(D .* x)));
        x = vec(x);
    end

end


function [x] = fastqsm_(f, mask, vsz, D)

    f = fft3(f);

    % Equation (8)
    x = sign(D) .* f;

    % Equation (10)
    pa = 1;
    pb = 30;
    n = 0.001;

    wfs = dipoleKspaceWeightsIlsqr(D, n, pa, pb);

    % Equation (9)
    r = 3;

    h = smvKernel(size(f), r, vsz, class(f));
    h = real(fft3(ifftshift(h)));

    x = fft3(mask .* real(ifft3(wfs .* x + (1-wfs) .* (h .* x))));

    % Equation (11)
    x = mask .* real(ifft3(wfs .* x + (1-wfs) .* (h .* x)));

    % Equation (12)
    t0 = 1/8;

    I = abs(D) < t0;

    iD = 1 ./ D;
    iD(I) = sign(D(I)) ./ t0;

    xtkd = mask .* real(ifft3(iD .* f));

    % Equation (13)
    A = [vec(x), ones(numel(x), 1, class(x))];

    ab = A \ vec(xtkd);

    % Equation (14)
    x = ab(1) .* x + ab(2);
    x = mask .* x;

end


function [xsa] = susceptibilityArtifacts_(x0, xfs, mask, vsz, D, tol, maxit, verbose)

    % Equation (15)
    pmin = 50;
    pmax = 70;
    [wx, wy, wz] = gradientWeightsIlsqr(xfs, mask, vsz, pmin, pmax);

    % Equation (4)
    thr = 0.1;
    Mic = abs(D) < thr;

    % Equation (3)
    [dx, dy, dz] = grad_(x0, vsz);

    bx = wx .* dx;
    by = wy .* dy;
    bz = wz .* dz;

    b = cat(4, bx, by, bz);

    sz3 = size(D);
    sz4 = size(b);

    xsa = lsmr(@afun, vec(b), [], tol, tol, [], maxit, [], verbose);
    xsa = mask .* reshape(xsa, sz3);

    function x = afun(x, t)
        if t(1) ~= 1 || t(1) == 't'
            x = reshape(x, sz4);
            x = gradAdj_(wx.*x(:,:,:,1), wy.*x(:,:,:,2), wz.*x(:,:,:,3), vsz);
            x = real(ifft3(Mic .* fft3(x)));
        else
            x = reshape(x, sz3);
            x = real(ifft3(Mic .* fft3(x)));
            [dx, dy, dz] = grad_(x, vsz);
            x = cat(4, wx.*dx, wy.*dy, wz.*dz);
        end
        x = vec(x);
    end

end


function [dx, dy, dz] = grad_(u, h)

    dx = zeros(size(u), 'like', u);
    dy = zeros(size(u), 'like', u);
    dz = zeros(size(u), 'like', u);

    gradfp_mex(dx, dy, dz, u, h);

    %ih = 1 ./ h;

    %dx = circshift(u, [-1,0,0]) - u;
    %dx = ih(1) .* dx;

    %dy = circshift(u, [0,-1,0]) - u;
    %dy = ih(2) .* dy;

    %dz = circshift(u, [0,0,-1]) - u;
    %dz = ih(3) .* dz;

end


function [d2u] = gradAdj_(dx, dy, dz, h)

    d2u = zeros(size(dx), 'like', dx);
    gradfp_adj_mex(d2u, dx, dy, dz, h);

    %ih = -1 ./ h;

    %dx = dx - circshift(dx, [1,0,0]);
    %dx = ih(1) .* dx;

    %dy = dy - circshift(dy, [0,1,0]);
    %dy = ih(2) .* dy;

    %dz = dz - circshift(dz, [0,0,1]);
    %dz = ih(3) .* dz;

    %d2u = dx + dy + dz;

end
