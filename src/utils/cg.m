function [x, ch] = cg(A, b, x, Pl, tol, maxit, verbose)
%CG Preconditioned Conjugate Gradient Method.
%   Attemps to solve the linear system A*x = b for x. Unlike MATLAB's pcg this
%   method works with n-dimensional inputs.
%
%   [x, ch] = CG(A, b, x, [Pl], [tol], [maxit], [verbose]);
%
%   Inputs
%   ------
%       A           function handle for computing A*x.
%       b           right hand side n-dimensional array.
%       x           initial guess.
%
%       Pl          function handle for symmetric positive definite
%                   preconditioner.
%                   default = []
%       tol         stopping tolerance.
%                   default = sqrt(eps(class(b)))
%       maxit       maximum number of iterations.
%                   default = numel(x)
%       verbose     boolean flag for printing information each iteration.
%                   default = false
%
%   Outputs
%   -------
%       x           solution to A*x = b.
%       ch          struct containing convergence history.
%
%   See also PCG

    narginchk(3, 7);

    if nargin < 7, verbose = false; end
    if nargin < 6 || isempty(maxit), maxit = numel(x); end
    if nargin < 5 || isempty(tol), tol = sqrt(eps(class(b))); end
    if nargin < 4 || isempty(Pl), Pl = []; end

    if ~isa(A, 'function_handle')
        error('A must be a function handle.');
    end

    if ~isempty(Pl) && ~isa(Pl, 'function_handle')
        error('Pl must be a function handle.');
    end


    ch = struct();
    ch.tol = tol;
    ch.resnorm = [];
    ch.isconverged = false;
    ch.iters = 0;

    p = zeros(size(x), 'like', x);
    r = b;
    q = zeros(size(x), 'like', x);

    if any(vec(x))
        q = A(x);
        r = r - q;
        res = norm2(r);
        reltol = norm2(b) * tol;
    else
        res = norm2(b);
        reltol = res * tol;
    end

    if isempty(Pl)
        [x, ch] = cg_(A, x, r, q, p, reltol, res, 1, maxit, verbose, ch);
    else
        [x, ch] = pcg_(Pl, A, x, r, q, p, reltol, res, 1, maxit, verbose, ch);
    end

end


function [x, ch] = cg_(A, x, r, q, p, reltol, res, prevres, maxit, verbose, ch)

    if verbose
        fprintf('iter%6s\tresidual\n', '');
    end

    for ii = 1:maxit
        if res <= reltol
            ch.isconverged = true;
            break;
        end

        p = r + res^2/prevres^2 .* p;
        q = A(p);
        a = res^2 / (vec(p)' * vec(q));

        x = x + a .* p;
        r = r - a .* q;

        prevres = res;
        res = norm2(r);

        ch.iters = ch.iters + 1;
        ch.resnorm = [ch.resnorm, res];

        if verbose
            fprintf('%5d/%d\t%1.3e\n', ii, maxit, res);
        end
    end

    if verbose
        fprintf('\n');
    end

end


function [x, ch] = pcg_(Pl, A, x, r, q, p, reltol, res, rho, maxit, verbose, ch)

    if verbose
        fprintf('iter%6s\tresidual\n', '');
    end

    for ii = 1:maxit
        if res <= reltol
            ch.isconverged = true;
            break;
        end

        prevrho = rho;

        q = Pl(r);
        rho = vec(r)' * vec(q);

        beta = rho / prevrho;
        p = q + beta .* p;

        q = A(p);
        a = rho / (vec(p)' * vec(q));

        x = x + a .* p;
        r = r - a .* q;

        res = norm2(r);

        ch.iters = ch.iters + 1;
        ch.resnorm = [ch.resnorm, res];

        if verbose
            fprintf('%5d/%d\t%1.3e\n', ii, maxit, res);
        end
    end

    if verbose
        fprintf('\n');
    end

end
