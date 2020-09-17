function [x] = ndi(f, mask, vsz, w, bdir, alpha, tol, maxit, verbose)
%NDI Nonlinear dipole inversion.
%
%   [x] = NDI(f, mask, vsz, [w], [bdir], [alpha], [tol], [maxit], [verbose]);
%
%   Inputs
%   ------
%       f           unwrapped local field/phase (3d/4d array).
%       mask        binary mask of region of interest (3d array).
%       vsz         voxel size.
%
%       w           weights for data consistency.
%                   default: mask
%       bdir        unit vector of B field direction.
%                   default: [0, 0, 1]
%       alpha       step size for gradient descent. if alpha <= 0 or alpha = []
%                   then backtracking line search is used to find alpha.
%                   default: 1
%       tol         stopping tolerance.
%                   || gradF || < tol
%                   default: sqrt(eps(class(f)))
%       maxit       maximum number of iterations.
%                   default: 50
%       verbose     boolean flag for printing information each iteration.
%                   default: true
%
%   Outputs
%   -------
%       x           susceptibility map.
%
%   Notes
%   -----
%       This method uses early stopping for regularization, ie. `maxit` is
%       the regularization parameter.
%
%   References
%   ----------
%       [1] Polak D, Chatnuntawech I, Yoon J, Iyer SS, Milovic C, Lee J,
%       Bachert P, Adalsteinsson E, Setsompop K, Bilgic B. Nonlinear dipole
%       inversion (NDI) enables robust quantitative susceptibility mapping
%       (QSM). NMR in Biomedicine. 2020 Feb 20:e4271.

    narginchk(3, 9);

    if nargin < 9, verbose = true; end
    if nargin < 8 || isempty(maxit), maxit = 150; end
    if nargin < 7 || isempty(tol), tol = sqrt(eps(class(f))); end
    if nargin < 6 || isempty(alpha), alpha = 1; end
    if nargin < 5 || isempty(bdir), bdir = [0, 0, 1]; end
    if nargin < 4 || isempty(w), w = mask; end


    % output
    x = zeros(size(f), 'like', f);

    D = dipoleKernel(size(mask), vsz, bdir, 'k', class(f));

    % 4d multi-echo loop
    for t = 1:size(f, 4)
        if verbose && size(f, 4) > 1
            fprintf('Echo: %d/%d\n', t, size(f, 4));
        end

        if size(w, 4) == 1
            w_ = w;
        else
            w_ = w(:,:,:,t);
        end

        x(:,:,:,t) = ndi_(f(:,:,:,t), w_, mask, D, alpha, tol, maxit, verbose);
    end

end


function [x] = ndi_(f, w, m, D, alpha, tolp, maxit, verbose)

    % step-size tolerance for linesearch
    tola = 100*eps(class(f));

    % pre-compute
    w2 = 2 .* w .* w;

    x = zeros(size(f), 'like', f);

    if ~isempty(alpha) && alpha > 0
        linesearch = 0;
    else
        linesearch = 1;
        ff = exp(1i .* f);
        F = @(x) norm2(w .* (ff - exp(1i.*(real(ifft3(D .* fft3(x)))))))^2;
        Fx = F(x);
    end

    if verbose
        fprintf('\niter\t\t||gradF||\n');
    end

    for ii = 1:maxit
        p = real(ifft3(D .* fft3(x))) - f;
        p = real(ifft3(D .* fft3(w2 .* sin(p))));

        pnrm = norm2(p);

        if ~linesearch
            x = x - alpha .* p;

        else
            alpha = 1;
            C = 0.5 * pnrm^2;
            while true
                x1 = x - alpha .* p;
                Fx1 = F(x1);

                if Fx - Fx1 >= alpha * C || alpha < tola
                    Fx = Fx1;
                    x = x1;
                    break;
                end

                alpha = 0.5 *alpha;
            end
        end

        if verbose
            fprintf('%3d/%d\t\t%.4e\n', ii, maxit, pnrm);
        end

        if pnrm <= tolp || alpha < tola
            break;
        end
    end

    if verbose
        fprintf('\n');
    end

    x = m .* x;

end
