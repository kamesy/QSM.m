function [] = testPoissonSolvers()
% Test different solvers for the dirichlet problem

    fprintf('***** Testing Poisson Solvers *****\n\n');

    for n = [63, 127, 255, 64, 128, 256]

        %sz = [2*n,2*n,n] + 1;
        sz = [n,n,n] + 1;
        h = 1./(sz-1);

        [V, G, d2u, d2uh] = modelProblem(h, 2*pi);
        f = d2u;

        fprintf('\tsize = %dx%dx%d \th = 1/%d\n\n', size(f), n);

        [v1, t1] = run_vcycle(f, G, h);
        [v2, t2] = run_fmg(f, G, h);
        [v3, t3] = run_mgpcg(f, G, h);

        printErr(v1, V, f, G, h, t1, 'V-cycle ');
        printErr(v2, V, f, G, h, t2, 'Fmg     ');
        printErr(v3, V, f, G, h, t3, 'MGPCG   ');
        fprintf('\n');
    end

end


function [v, t] = run_vcycle(f, G, h)

    o.tol = sqrt(eps(class(f)));
    o.maxit = max(size(f));
    o.mu = 1;
    o.npre = 1;
    o.npost = 1;
    o.nboundary = 2;
    o.nlevels = max(1, floor(log2(min(size(f))))-3);

    fprintf('\tRunning V-cycle\n');
    %disp(o)

    v = zeros(size(f), 'like', f);
    tic;
        v = mg(v, f, G, h, o);
    t = toc;

end


function [v, t] = run_fmg(f, G, h)

    o.tol = sqrt(eps(class(f)));
    o.maxit = 1;
    o.mu = 1;
    o.npre = 1;
    o.npost = 1;
    o.nboundary = 2;
    o.nlevels = max(1, floor(log2(min(size(f))))-3);

    fprintf('\tRunning Full multigrid\n');

    tic;
        v = fmg(f, G, h, o);
    t = toc;

end


function [v, t] = run_mgpcg(f, G, h)

    tol = sqrt(eps(class(f)));
    maxit = numel(f);
    verbose = true;

    o.tol = -1; %sqrt(eps(class(f)));
    o.maxit = 1; %max(size(f));
    o.mu = 1;
    o.npre = 1;
    o.npost = 1;
    o.nboundary = 2;
    o.nlevels = max(1, floor(log2(min(size(f))))-3);

    fprintf('\tRunning MGPCG\n');
    %disp(o)

    tic;
        v = mgpcg(f, G, h, tol, maxit, o, verbose);
    t = toc;

end


function [] = printErr(v, V, f, G, h, t, str)

    r = L2norm(residual_(f, v, G, h));
    e = L2norm(v-V);

    fprintf('\t\t\t%s \t||r||_L2 = %.2e \t||e||_L2 = %.2e \t time = %.3fs\n', ...
        str, r, e, t);

end


function [u, G, d2u, d2uh] = modelProblem(h, p)

    if nargin < 2, p = pi; end

    [X, Y, Z] = ndgrid(0:h(1):1, 0:h(2):1, 0:h(3):1);

    u = sin(p*X) .* sin(p*Y) .* sin(p*Z);

    G = false(size(u));
    G(2:end-1,2:end-1,2:end-1) = 1;

    d2u = -3*p^2*u;
    d2uh = lapmg_(u, G, h);

    d2u = -d2u;
    d2uh = -d2uh;

end


function [e] = L2norm(u)
    e = norm(reshape(u, [], 1))/sqrt(numel(u));
end


function [G2] = coarsen_grid_(G)
    G2 = false(floor((size(G)-2+1)/2)+2);
    coarsen_grid_mex(G2, G);
end

function [p] = restrict_(r, G)
    p = zeros(size(G), 'like', r);
    restrict_mex(p, r, G);
end

function [p] = prolong_(v, G)
    p = zeros(size(G), 'like', v);
    prolong_mex(p, v, G);
end

function [v] = correct_(v, v2, G)
    correct_mex(v, v2, G);
end

function [r] = residual_(f, v, G, h)
    r = zeros(size(f), 'like', f);
    residual_mex(r, f, v, G, h);
end

function [v] = gauss_seidel_(v, f, G, h, ii, rev)
    gauss_seidel_mex(v, f, G, h, ii, rev);
end

function [l] = lapmg_(v, G, h)
    l = zeros(size(v), 'like', v);
    lapmg_mex(l, v, G, h);
end

function [B] = boundary_mask_(G)
    B = false(size(G));
    boundary_mask_mex(B, G);
end
