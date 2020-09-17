function [] = testMultigridPoisson(n)
% Testing multigrid
%
%   References
%   ----------
%       [1] Briggs WL, Henson VE, McCormick SF. A multigrid tutorial,
%       Second Edition. Siam; 2000.

    if n == 1
        testFixedPointRelaxation();

    elseif n == 2
        testFixedPointVcycle();

    elseif n == 3
        testTwoLevelCycle();

    elseif n == 4
        testVcycle();

    end

end


function [] = testVcycle()
% Smoothing factor for red-black gauss seidel for 3d poisson is: 0.567
%   Example 3.1 in [1]
%
%   [1] Hocking LR, Greif C. Closed-form multigrid smoothing factors for
%       lexicographic Gauss–Seidel. IMA Journal of Numerical Analysis.
%       2011 Nov 4;32(3):795-812.

    fprintf('***** Testing V-cycle *****\n\n');

    for n = [15, 16, 31, 32, 63, 64, 127, 128, 255, 256]

        sz = [n,n,n] + 1;
        h = 1./(sz-1);

        [V, G, d2u, d2uh] = modelProblem(h, 2*pi);
        f = d2u;

        fprintf('\t\th = 1/%d\tsize = %dx%dx%d\n\n', n, size(f));

        G = padarray(G, [1,1,1]);
        f = padarray(f, [1,1,1]);
        V = padarray(V, [1,1,1]);

        v = zeros(size(f), 'like', f);
        r = residual_(f, v, G, h);

        Lr = L2norm(r);
        err = L2norm(v - V);

        fprintf('\t\t\tIter %3d \t||r||_L2 = %.2e \t||e||_L2 = %.2e\n\n', ...
            0, Lr(1), err(1));

        nlevels = max(1, floor(log2(n))-3);

        tic;
        for ii = 1:1000
            v = vcycle(v, f, G, h, 1, 1, nlevels);
            r = residual_(f, v, G, h);
            Lr(ii+1) = L2norm(r);
            err(ii+1) = L2norm(v-V);

            fprintf(sprintf('\t\t\tIter %3d \t||r||_L2 = %.2e \tratio = %.2f \t||e||_L2 = %.2e \tratio = %.2f\n', ...
                ii, Lr(ii+1), Lr(ii+1)/Lr(ii), err(ii+1), err(ii+1)/err(ii)));

            if Lr(ii+1) < sqrt(eps(class(f)))*L2norm(f)
                break
            end
        end
        fprintf('\n');
        toc;
    end

end


function [] = testTwoLevelCycle()
% Test intergrid operators. Residual/error should be lower on up relaxation
%   than down relaxation each cycle. Coarse grid should be solved accurately.

    fprintf('***** Testing Two Level Cycle *****\n\n');

    for n = [63, 64]

        sz = [n,n,n] + 1;
        h = 1./(sz-1);

        fprintf('\t\th = 1/%d\tsize = %dx%dx%d\n', n, sz);

        [V, G, d2u, d2uh] = modelProblem(h, 2*pi);
        f = d2u;

        G = padarray(G, [1,1,1]);
        f = padarray(f, [1,1,1]);
        V = padarray(V, [1,1,1]);

        G2 = coarsen_grid_(G);

        v = zeros(size(f), 'like', f);
        r = residual_(f, v, G, h);

        Lrd = L2norm(r);
        erd = L2norm(v - V);

        Lru = Lrd(1);
        eru = erd(1);

        fprintf('\t\t\tIter %3d \t||r||_L2 = %.2e \t||e||_L2 = %.2e\n\n', ...
            0, Lru(1), eru(1));

        tic;
        for ii = 1:500

            v = gauss_seidel_(v, f, G, h, 2, 0);

            rd = residual_(f, v, G, h);

            Lrd(ii+1) = L2norm(rd);
            erd(ii+1) = L2norm(v-V);

            f2 = restrict_(rd, G2);

            v2 = zeros(size(f2), 'like', f2);
            while true
                v2 = gauss_seidel_(v2, f2, G2, 2*h, 8, 0);
                v2 = gauss_seidel_(v2, f2, G2, 2*h, 8, 1);
                r2 = residual_(f2, v2, G2, 2*h);
                nr2 = L2norm(r2);
                if nr2 < eps(class(r2))*numel(r2)
                    break;
                end
            end

            v = correct_(v, v2, G);
            v = gauss_seidel_(v, f, G, h, 2, 1);

            ru = residual_(f, v, G, h);

            Lru(ii+1) = L2norm(ru);
            eru(ii+1) = L2norm(v-V);

            fprintf(sprintf('\t\t\tIter %3d \t||r||_L2 = %.2e / %.2e \tratio = %.2f / %.2f / %.2f \t||e||_L2 = %.2e / %.2e \tratio = %.2f / %.2f / %.2f\n', ...
                ii, Lrd(ii+1), Lru(ii+1), Lrd(ii+1)/Lrd(ii), Lru(ii+1)/Lru(ii), Lru(ii+1)/Lrd(ii+1), ...
                erd(ii+1), eru(ii+1), erd(ii+1)/erd(ii), eru(ii+1)/eru(ii), eru(ii+1)/erd(ii+1)));

            if Lru(ii+1) < eps(class(ru))*numel(ru)
                break;
            end
        end
        fprintf('\n');
        toc;
    end

end


function [] = testFixedPointVcycle()
% Relaxation should not alter the exact solution.
%   Use exact solution as initial guess. Should get zero residual before and
%   after relaxation.

    fprintf('***** Testing Fixed Point Property: V-cycle *****\n\n');

    for n = [31, 32, 126, 127]
        sz = [n,n,n] + 1;
        h = 1./[n,n,n];

        fprintf('\t\th = 1/%d\tsize = %dx%dx%d\n', n, sz);

        V = randn(sz+2);
        G = false(sz+2);

        G(2:end-1, 2:end-1, 2:end-1) = 1;
        V = G .* V;

        f = -lapmg_(V, G, 1./[n,n,n]);

        v = V;
        r = residual_(f, v, G, h);

        Lrd = L2norm(r);
        Lru = Lrd(1);

        fprintf('\t\t\tInitial    \t||e||_L2 = %.2e \t\n\n', L2norm(v-V));

        f = {f};
        v = {v};
        G = {G};
        r = {r};

        nlevels = floor(log2(n))-2;

        tic;
        for l = 1:nlevels
            v{l} = gauss_seidel_(v{l}, f{l}, G{l}, 2^(l-1)*h, 3, 0);
            r{l+1} = residual_(f{l}, v{l}, G{l}, 2^(l-1)*h);

            G{l+1} = coarsen_grid_(G{l});
            f{l+1} = restrict_(r{l+1}, G{l+1});
            v{l+1} = zeros(size(G{l+1}), 'like', v{l});

            Lrd(l+1) = L2norm(r{l+1});

            fprintf(sprintf('\t\t\t%sLevel %d \t||r||_L2 = %.2e\n', ...
                repmat(' ', [1, l-1]), l-1, Lrd(l+1)));
        end

        v{end} = gauss_seidel_(v{end}, f{end}, G{end}, 2^nlevels*h, 2500, 0);
        v{end} = gauss_seidel_(v{end}, f{end}, G{end}, 2^nlevels*h, 2500, 1);
        r{nlevels+1} = residual_(f{end}, v{end}, G{end}, 2^nlevels*h);

        Lrd(nlevels+1) = L2norm(r{nlevels+1});

        fprintf(sprintf('\t\t\t%sLevel %d \t||r||_L2 = %.2e\n', ...
            repmat(' ', [1, nlevels]), nlevels, Lrd(nlevels+1)));

        for l = nlevels:-1:1
            v{l} = correct_(v{l}, v{l+1}, G{l});
            v{l} = gauss_seidel_(v{l}, f{l}, G{l}, 2^(l-1)*h, 3, 1);

            rr = residual_(f{l}, v{l}, G{l}, 2^(l-1)*h);
            Lru(l+1) = L2norm(rr);

            fprintf(sprintf('\t\t\t%sLevel %d \t||r||_L2 = %.2e\n', ...
                repmat(' ', [1, l-1]), l-1, Lru(l+1)));
        end
        fprintf('\n');
        fprintf('\t\t\tResult   \t||e||_L2 = %.2e\n', L2norm(v{1} - V));
        fprintf('\n');
        toc;
    end

end


function [] = testFixedPointRelaxation()
% Relaxation should not alter the exact solution.
%   Use exact solution as initial guess. Should get zero residual before and
%   after relaxation.

    fprintf('***** Testing Fixed Point Property: Relaxation *****\n\n');

    for s = [0, 1]
        if s == 0
            fprintf('\tGauss-Seidel forward\n');
        else
            fprintf('\tGauss-Seidel backward\n');
        end

        for n = [15, 16, 127, 128]

            sz = [n,n,n] + 1;
            h = 1./[n,n,n];

            fprintf('\t\th = 1/%d\tsize = %dx%dx%d\n', n, sz);

            G = true(sz);
            V = randn(sz);

            f = -lapmg_(V, G, 1./[n,n,n]);

            v = V;
            r = residual_(f, v, G, h);

            Lr = L2norm(r);
            err = L2norm(v - V);

            fprintf('\t\t\tIter %3d \t||r||_L2 = %.2e \t||e||_L2 = %.2e\n\n', ...
                0, Lr(1), err(1));

            tic;
            for ii = 1:100

                v = gauss_seidel_(v, f, G, h, 1, s);
                r = residual_(f, v, G, h);

                Lr(ii+1) = L2norm(r);
                err(ii+1) = L2norm(v-V);

                if ii == 1 || mod(ii, 50) == 0
                    fprintf(sprintf('\t\t\tIter %3d \t||r||_L2 = %.2e \tratio = %.2f \t||e||_L2 = %.2e \tratio = %.2f\n', ...
                        ii, Lr(ii+1), Lr(ii+1)/Lr(ii), err(ii+1), err(ii+1)/err(ii)));
                end
            end
            fprintf('\n');
            toc;
        end
        fprintf('\n');
    end

end


function [v] = vcycle(v, f, G, h, ipre, ipost, nlevels)

    v = {v};
    f = {f};
    G = {G};
    B = {boundary_mask_(G{1})};

    for l = 1:nlevels
        v{l} = gauss_seidel_(v{l}, f{l}, G{l}, 2^(l-1)*h, ipre, 1);
        v{l} = gauss_seidel_(v{l}, f{l}, B{l}, 2^(l-1)*h, 2^(l-1)*2, 1);

        r = residual_(f{l}, v{l}, G{l}, 2^(l-1)*h);

        G{l+1} = coarsen_grid_(G{l});
        B{l+1} = boundary_mask_(G{l+1});
        f{l+1} = restrict_(r, G{l+1});
        v{l+1} = zeros(size(G{l+1}), 'like', v{l});
    end

    reltol = sqrt(eps(class(r)))*norm(vec(f{l}.*G{l}));

    for ii = 1:1024
        v{end} = gauss_seidel_(v{end}, f{end}, G{end}, 2^nlevels*h, 128, 1);
        v{end} = gauss_seidel_(v{end}, f{end}, G{end}, 2^nlevels*h, 128, 0);
        r = residual_(f{end}, v{end}, G{end}, 2^nlevels*h);
        p = norm(vec(r.*G{end}));
        if p < reltol
            break
        end
    end

    for l = nlevels:-1:1
        v{l} = correct_(v{l}, v{l+1}, G{l});

        v{l} = gauss_seidel_(v{l}, f{l}, B{l}, 2^(l-1)*h, 2^(l-1)*2, 0);
        v{l} = gauss_seidel_(v{l}, f{l}, G{l}, 2^(l-1)*h, ipost, 0);
    end

    v = v{1};

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
