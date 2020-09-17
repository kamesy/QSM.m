function [x] = medi(f, mask, vsz, mag, w, bdir, opts)
%MEDI Morphology enabled dipole inversion
%
%   [x] = MEDI(f, mask, vsz, [mag], [w], [bdir], [opts]);
%
%   Inputs
%   ------
%       f               unwrapped local field/phase (3d/4d array).
%       mask            binary mask of region of interest (3d array).
%       vsz             voxel size.
%
%       mag             anatomical image for gradient mask.
%                       default: ones(size(mask))
%       w               weights for data consistency.
%                       default: ones(size(mask))
%       bdir            unit vector of B field direction.
%                       default: [0, 0, 1]
%       opts.lambda     regularization parameter.
%                       default: 7.5e-5
%       opts.gpct       percentage of voxels considered to be edges for
%                       gradient mask.
%                       default: 30
%       opts.tol        stopping tolerance for gauss newton method.
%                       ||x_{k+1} - x_{k}||_2 <= tol * ||x_{k}||_2
%                       default: 1e-1
%       opts.maxit      maximum number of iterations of gauss newton method.
%                       default: 30
%       opts.tolcg      stopping tolerance for conjugate gradient.
%                       default: 1e-1
%       opts.maxitcg    maximum number of iterations of conjugate gradient.
%                       default: 10
%       opts.merit      boolean flag for Model Error Reduction through
%                       Iterative Tuning.
%                       default: false
%       opts.verbose    boolean flag for printing information each iteration.
%                       default: true
%
%   Outputs
%   -------
%       x               susceptibility map.
%
%   References
%   ----------
%       [1] Liu T, Wisnieff C, Lou M, Chen W, Spincemaille P, Wang Y. Nonlinear
%       formulation of the magnetic field to source relationship for robust
%       quantitative susceptibility mapping. Magnetic resonance in medicine.
%       2013 Feb;69(2):467-76.

    narginchk(3, 7);

    if nargin < 7, opts = []; end
    if nargin < 6 || isempty(bdir), bdir = [0, 0, 1]; end
    if nargin < 5 || isempty(w), w = ones(size(mask), 'like', f); end
    if nargin < 4 || isempty(mag), mag = ones(size(mask), 'like', f); end

    if ~isfield(opts, 'lambda') || isempty(opts.lambda)
        opts.lambda = 7.5e-5;
    end

    if ~isfield(opts, 'gpct') || isempty(opts.gpct)
        opts.gpct = 30;
    end

    if ~isfield(opts, 'tol') || isempty(opts.tol)
        opts.tol = 1e-1;
    end

    if ~isfield(opts, 'maxit') || isempty(opts.maxit)
        opts.maxit = 30;
    end

    if ~isfield(opts, 'tolcg') || isempty(opts.tolcg)
        opts.tolcg = 1e-1;
    end

    if ~isfield(opts, 'maxitcg') || isempty(opts.maxitcg)
        opts.maxitcg = 10;
    end

    if ~isfield(opts, 'merit') || isempty(opts.merit)
        opts.merit = false;
    end

    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = true;
    end


    % output
    x = zeros(size(f), 'like', f);

    D = dipoleKernel(size(mask), vsz, bdir, 'k', class(f));

    % 4d multi-echo loop
    for t = 1:size(f, 4)
        if opts.verbose && size(f, 4) > 1
            fprintf('Echo: %d/%d\n', t, size(f, 4));
        end

        if size(w, 4) == 1
            w_ = w;
        else
            w_ = w(:,:,:,t);
        end

        if size(mag, 4) == 1
            m_ = mag;
        else
            m_ = mag(:,:,:,t);
        end

        [mx, my, mz] = gradientMaskMedi(m_, mask, vsz, opts.gpct);

        if ~any(vec(mx))
            mx = m_;
        end
        if ~any(vec(my))
            my = m_;
        end
        if ~any(vec(mz))
            mz = m_;
        end

        m_ = cat(4, mx, my, mz);

        x(:,:,:,t) = medi_(f(:,:,:,t), m_, w_, mask, D, vsz, opts);
    end

end


function [x] = medi_(f, m, w, mask, D, vsz, opts)

    lam = opts.lambda;
    merit = opts.merit;

    tol = opts.tol;
    maxit = opts.maxit;
    tolcg = opts.tolcg;
    maxitcg = opts.maxitcg;

    verbose = opts.verbose;

    beta = sqrt(eps(class(f)));

    % pre-compute
    w2 = w .* w;

    x = zeros(size(f), 'like', f);

    Dx = real(ifft3(D .* fft3(x)));
    xnrm = norm2(x);

    if verbose
        fprintf('\niter\t\t||dx||/||x||\n');
    end

    for ii = 1:maxit

        mx = m(:,:,:,1);
        my = m(:,:,:,2);
        mz = m(:,:,:,3);

        [ux, uy, uz] = grad_(x, vsz);

        ux = mx .* ux;
        uy = my .* uy;
        uz = mz .* uz;

        P = 1 ./ sqrt(ux.*ux + uy.*uy + uz.*uz + beta);

        ux = mx .* P .* ux;
        uy = my .* P .* uy;
        uz = mz .* P .* uz;

        b = lam .* gradAdj_(ux, uy, uz, vsz);
        b = b + real(ifft3(conj(D) .* fft3(1i .* w2 .* (exp(1i.*(f - Dx)) - 1))));

        A = @(du) afun(du, P, w2, m, D, vsz, lam);

        dx = zeros(size(x), 'like', x);
        dx = cg(A, -b, dx, [], tolcg, maxitcg, 0);

        x = x + dx;

        dnrm = norm2(dx) / xnrm;
        xnrm = norm2(x);

        if verbose
            fprintf('%3d/%d\t\t%.4e\n', ii, maxit, dnrm);
        end

        if dnrm <= tol
            break;
        end

        % prepare for next round
        Dx = real(ifft3(D .* fft3(x)));

        if merit
            r = abs(w .* (exp(1i .* Dx) - exp(1i .* f)));
            r = r ./ (6 .* std(r(mask)));
            I = r >= 1;
            w(I) = w(I) ./ r(I).^2;
            w2 = w .* w;
        end

    end

    if verbose
        fprintf('\n');
    end

    x = mask .* x;

end


function [y] = afun(dx, P, w2, m, D, vsz, lam)

    mx = m(:,:,:,1);
    my = m(:,:,:,2);
    mz = m(:,:,:,3);

    [ux, uy, uz] = grad_(dx, vsz);

    ux = mx .* P .* mx .* ux;
    uy = my .* P .* my .* uy;
    uz = mz .* P .* mz .* uz;

    R = lam .* gradAdj_(ux, uy, uz, vsz);
    D = real(ifft3(conj(D) .* fft3(w2 .* real(ifft3(D .* fft3(dx))))));

    y = D + R;

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
