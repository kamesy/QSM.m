function [f, df, p, ch] = fitEchoNonlinear(phas, mag, TEs, tol, maxit, teUnwrap, verbose)
%FITECHONONLINEAR Non-linear fit for multi-echo data.
%   Gauss-Newton algorithm to fit multi-echo phase data. A weighted linear fit
%   is used as initial guess.
%
%   [f, df, p, ch] =
%       FITECHONONLINEAR(phas, [mag], [TEs], [tol], [maxit], [teUnwrap], [verbose]);
%
%   Notes
%   -----
%       This method is highly dependent on a good initial guess and is not
%       robust. If the result is not acceptable, as is the case for highly
%       wrapped phase data, try to unwrap and to remove background fields
%       before fitting.
%
%   References
%   ----------
%       [1] Liu T, Wisnieff C, Lou M, Chen W, Spincemaille P, Wang Y.
%       Nonlinear formulation of the magnetic field to source relationship for
%       robust quantitative susceptibility mapping. Magnetic resonance in
%       medicine. 2013 Feb;69(2):467-76.

    narginchk(1, 7);

    if nargin < 7, verbose = 0; end
    if nargin < 6 || isempty(teUnwrap), teUnwrap = 1; end
    if nargin < 5 || isempty(maxit), maxit = 500; end
    if nargin < 4 || isempty(tol), tol = 1e-3; end
    if nargin < 3 || isempty(TEs), TEs = 1:size(phas, 4); end
    if nargin < 2 || isempty(mag), mag = ones(size(phas)); end

    % use line search. NOT tested.
    LINESEARCH = 0;

    alpha = 1;  % step-size
    lambda = 0; % damping parameter

    T = class(phas);
    mag = cast(mag, T);
    TEs = cast(TEs, T);

    sz0 = size(mag);
    nTE = numel(TEs);

    TEs = reshape(TEs, 1, []);
    TE0 = ones(size(TEs), T);

    mag = mag ./ max(vec(mag));

    % temporal unwrap for initial guess
    if teUnwrap
        phas = temporalUnwrap(phas);
    end

    % compute initial guess
    [f, df, p] = fitEchoLinear(phas, mag, TEs);

    phas = reshape(phas, [], nTE);
    mag = reshape(mag, [], nTE);
    f = reshape(f, [], 1);
    p = reshape(p, [], 1);

    sz = size(phas);

    % pre-compute Re{Jt.J}^-1
    M = mag .* mag;
    a0 = sum(M .* (ones(sz(1), 1) * TEs.^2), 2) * (1 + lambda);
    bc = sum(M .* (ones(sz(1), 1) * TEs), 2);
    d0 = sum(M, 2) * (1 + lambda);

    deter = a0.*d0 - bc.*bc;
    a =  d0 ./ deter;
    bc = -bc ./ deter;
    d =  a0 ./ deter;

    clear a0 d0 deter;

    a(~isfinite(a)) = 0;
    bc(~isfinite(bc)) = 0;
    d(~isfinite(d)) = 0;

    % pre-compute part of Re{Jt.g(x)}
    A = 1i * [vec(TEs), vec(TE0)];

    % objective function for line search
    F = @(f, p) 2 .* sum(M .* (1 - cos(phas - f*TEs - p*TE0)), 2);

    % stopping criterion
    ch = zeros(maxit, 1);

    % compute gradF with initial guesses
    J = real((M .* (exp(1i*(phas - f*TEs - p*TE0)) - 1)) * A);
    reltol = tol * max(norm(vec(J), Inf), sqrt(eps(T)));

    for ii = 1:maxit
        % Re{Jt.g(x)}
        J = real((M .* (exp(1i*(phas - f*TEs - p*TE0)) - 1)) * A);

        Jf = J(:,1);
        Jp = J(:,2);

        % Re{Jt.J}^-1 * Re{Jt.g(x)}
        df = a.*Jf + bc.*Jp;
        dp = bc.*Jf + d.*Jp;

        if ~LINESEARCH
            f = f - alpha.*df;
            p = p - alpha.*dp;

        else
            % line search
            alpha = ones(size(f), T);
            C = 0.5 .* (df.*Jf + dp.*Jp);
            Fk = F(f, p);

            while true
                f1 = f - alpha.*df;
                p1 = p - alpha.*dp;

                I = Fk - F(f1, p1) >= alpha .* C;

                % `all` is aggressive. could try sum(I) > 0.85*numel(I)
                if all(I) || any(alpha < sqrt(eps(df)))
                    f = f1;
                    p = p1;
                    break;
                end

                alpha(~I) = 0.5 * alpha(~I);
            end
        end

        ch(ii) = norm(vec(J), Inf);

        if verbose
            fprintf([ ...
                'Iter: %d/%d, ||gradF||_inf: %g, ', ...
                '||df||_inf: %g, ||dp||_inf: %g\n', ...
            ], ii, maxit, ch(ii), norm(df, Inf), norm(dp, Inf));
        end

        if ch(ii) <= reltol || all(alpha < sqrt(eps(df)))
            ch(ii+1:end) = [];
            break;
        end
    end

    f = reshape(f, sz0(1:3));

    if nargout > 2
        p = reshape(p, sz0(1:3));
    end

end


function [uphas] = temporalUnwrap(phas)

    uphas = phas;
    for t = 1:size(phas, 4)-1
        dp = phas(:,:,:,t+1) - phas(:,:,:,t);

        I1 = dp <= -pi;
        I2 = dp > pi;
        dp(I1) = dp(I1) + 2*pi;
        dp(I2) = dp(I2) - 2*pi;

        uphas(:,:,:,t+1) = uphas(:,:,:,t) + dp;
    end

end
