function [x] = tv(f, mask, vsz, bdir, opts)
%TV Total variation deconvolution using ADMM.
%
%   [x] = TV(f, mask, vsz, [bdir], [opts]);
%
%   Inputs
%   ------
%       f               unwrapped local field/phase (3d/4d array).
%       mask            binary mask of region of interest (3d array).
%       vsz             voxel size.
%
%       bdir            unit vector of B field direction.
%                       default: [0, 0, 1]
%       opts.mu         regularization parameter for tv minimization.
%                       default: 7.5e3
%       opts.tv         anisotropic (1) or isotropic (2) total variation
%                       default: 1
%       opts.rtol       relative stopping tolerance
%                       default: 1e-1
%       opts.atol       absolute stopping tolerance
%                       default: 1e-4
%       opts.maxit      maximum number of iterations.
%                       default: 20
%       opts.verbose    boolean flag for printing information each iteration.
%                       default: true
%
%   Outputs
%   -------
%       x               susceptibility map.
%
%   References
%   ----------

    narginchk(3, 5);

    if nargin < 5, opts = []; end
    if nargin < 4 || isempty(bdir), bdir = [0, 0, 1]; end

    if ~isfield(opts, 'mu') || isempty(opts.mu)
        opts.mu = 7.5e3;
    end

    if ~isfield(opts, 'tv') || isempty(opts.tv)
        opts.tv = 1; % 1 anisotropic, 2 isotropic
    end

    if ~isfield(opts, 'rtol') || isempty(opts.rtol)
        opts.rtol = 1e-1;
    end

    if ~isfield(opts, 'atol') || isempty(opts.atol)
        opts.atol = 1e-4;
    end

    if ~isfield(opts, 'maxit') || isempty(opts.maxit)
        opts.maxit = 20;
    end

    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = 1;
    end

    if ~isfield(opts, 'rho') || isempty(opts.rho)
        opts.rho = 9;
    end

    % over-relaxation parameter
    if ~isfield(opts, 'alpha') || isempty(opts.alpha)
        opts.alpha = 1.5;
    end

    % admm parameter update parameters
    if ~isfield(opts, 'rtau') || isempty(opts.rtau)
        opts.rtau = 0.7;
    end

    if ~isfield(opts, 'stau') || isempty(opts.stau)
        opts.stau = 1e2;
    end

    if ~isfield(opts, 'gamma') || isempty(opts.gamma)
        opts.gamma = 3;
    end


    % output
    x = zeros(size(f), 'like', f);

    D = dipoleKernel(size(mask), vsz, bdir, 'k', class(f));

    % 4d multi-echo loop
    for t = 1:size(f, 4)
        if opts.verbose && size(f, 4) > 1
            fprintf('Echo: %d/%d\n', t, size(f, 4));
        end

        x(:,:,:,t) = tv_(f(:,:,:,t), mask, D, vsz, opts);
    end

end


function [u] = tv_(f, mask, D, vsz, opts)

    T = class(f);
    sz = size(f);

    % stopping tolerance
    rtol = opts.rtol;
    atol = opts.atol;

    atolp = sqrt(3*numel(f)) * atol;
    atold = sqrt(numel(f)) * atol;

    % regularization parameter
    mu = opts.mu;

    % admm parameters
    rho = opts.rho;
    alpha = opts.alpha;

    rtau = opts.rtau;
    stau = opts.stau;
    gamma = opts.gamma;

    mtv = opts.tv;

    % admm maxit
    maxit = opts.maxit;

    % print stuff
    verbose = opts.verbose;

    % pre-compute
    irho = 1 / rho;

    F = mu*conj(D) .* fft3(f);

    K = mu*abs(D).^2;

    L = -laplaceKernel(vsz);
    L = padarray(L, sz-[3,3,3], 'post');
    L = real(fft3(circshift(L, -[1,1,1])));

    iA = 1 ./ (K + rho*L);
    iA(~isfinite(iA)) = 0;

    % initialize admm variables
    u = f;

    [l1, l2, l3] = deal(zeros(sz, T));
    [v1, v2, v3] = grad_(u, vsz);

    nr = sum(sqrt(vec(v1.*v1 + v2.*v2 + v3.*v3)));

    flag = 0;

    if verbose
        fprintf('\niter\t\t||r||\t\teps_pri\t\t||s||\t\teps_dual\n');
    end

    for ii = 1:maxit

        % *************************************************************** %
        % u - subproblem
        % *************************************************************** %

        r1 = rho.*v1 - l1;
        r2 = rho.*v2 - l2;
        r3 = rho.*v3 - l3;

        r1 = gradAdj_(r1, r2, r3, vsz);
        b  = F + fft3(r1);

        u  = real(ifft3(iA .* b));

        % *************************************************************** %
        % v - subproblem
        % *************************************************************** %

        v1_ = v1;
        v2_ = v2;
        v3_ = v3;

        [dx, dy, dz] = grad_(u, vsz);

        % over-relaxation
        dx1 = alpha*dx + (1-alpha)*v1;
        dy1 = alpha*dy + (1-alpha)*v2;
        dz1 = alpha*dz + (1-alpha)*v3;

        r1 = dx1 + irho*l1;
        r2 = dy1 + irho*l2;
        r3 = dz1 + irho*l3;

        if mtv == 1
            % anisotropic tv
            v1 = max(abs(r1)-irho, 0) .* sign(r1);
            v2 = max(abs(r2)-irho, 0) .* sign(r2);
            v3 = max(abs(r3)-irho, 0) .* sign(r3);
        else
            % isotropic tv
            s = sqrt(r1.*r1 + r2.*r2 + r3.*r3);
            s = (s - irho) .* (s > irho) ./ s;
            s(~isfinite(s)) = 0;

            v1 = r1 .* s;
            v2 = r2 .* s;
            v3 = r3 .* s;
        end

        % *************************************************************** %
        % Lagrange multiplier update
        % *************************************************************** %

        l1 = l1 + rho.*(dx1 - v1);
        l2 = l2 + rho.*(dy1 - v2);
        l3 = l3 + rho.*(dz1 - v3);

        % *************************************************************** %
        % convergence check
        % *************************************************************** %

        nr_ = nr;

        % primal residual
        r1 = dx - v1;
        r2 = dy - v2;
        r3 = dz - v3;

        % dual residual
        s = rho .* gradAdj_(v1-v1_, v2-v2_, v3-v3_, vsz);

        nr = sum(sqrt(vec(r1.*r1 + r2.*r2 + r3.*r3)));
        ns = norm2(s);

        na = sum(sqrt(vec(dx.*dx + dy.*dy + dz.*dz)));
        nb = sum(sqrt(vec(v1.*v1 + v2.*v2 + v3.*v3)));

        ep = atolp + rtol*max(na, nb);
        ed = atold + rtol*norm2(rho*gradAdj_(l1, l2, l3, vsz));

        if verbose
            fprintf('%3d/%d\t\t%.4e\t%.4e\t%.4e\t%.4e\n', ...
                ii, maxit, nr, ep, ns, ed);
        end

        if nr < ep && ns < ed
            break
        end

        % *************************************************************** %
        % parameter update
        % *************************************************************** %

        if nr > stau*ns || (nr > rtau*nr_ && flag < 3)
            rho  = gamma .* rho;
            irho = 1 / rho;

            iA = 1 ./ (K + rho*L);
            iA(~isfinite(iA)) = 0;

            if flag > 5
                flag = 0;
            end

        elseif nr < ns/stau
            rho  = rho ./ gamma^2;
            irho = 1 / rho;

            iA = 1 ./ (K + rho*L);
            iA(~isfinite(iA)) = 0;

            flag = flag + 1;
        end

    end

    if verbose
        fprintf('\n');
    end

    u = mask .* u;

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
