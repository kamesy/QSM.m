function [f, df, p0] = fitEchoLinear(phas, mag, TEs)
%FITECHOLINEAR Weighted linear fit for multi-echo data.
%   Weights are equal to the reciprocal of the variance of the phase which is
%   computed based on [1].
%
%   [f, df, p0] = FITECHOLINEAR(phas, [mag], [TEs]);
%
%   References
%   ----------
%       [1] Conturo TE, Smith GD. Signal‐to‐noise in phase angle
%       reconstruction: dynamic range extension using phase reference offsets.
%       Magnetic Resonance in Medicine. 1990 Sep;15(3):420-37.

    narginchk(1, 3);

    if nargin < 3, TEs = 1:size(phas, 4); end
    if nargin < 2 || isempty(mag), mag = ones(size(phas), 'like', phas); end


    TEs = reshape(TEs, 1, 1, 1, length(TEs));
    TEs = repmat(TEs, [size(phas(:,:,:,1)), 1]);
    w = mag .* mag;

    W = sum(w, 4);

    x = sum(w .* TEs , 4) ./ W;
    x = bsxfun(@minus, TEs, x);

    y = sum(w .* phas, 4) ./ W;
    y = bsxfun(@minus, phas, y);

    f = sum(w .* x .* y, 4);
    f = f ./ sum(w .* x .* x, 4);

    f(~isfinite(f)) = 0;

    % variance estimate
    if nargout > 1
        idphas = real(mag .* exp(1i*phas));
        idphas = var(reshape(idphas, [], size(TEs, 4)));
        idphas = bsxfun(@rdivide, w, reshape(idphas, 1, 1, 1, size(TEs, 4)));

        a0 = sum(idphas, 4);
        bc = sum(idphas .* TEs, 4);
        d0 = sum(idphas .* TEs.^2, 4);

        df = a0 ./ (a0 .* d0 - bc.^2);
        df(~isfinite(df)) = 0;
    end

    % phas(TE = 0)
    if nargout > 2
        x = sum(w .* TEs , 4) ./ W;
        y = sum(w .* phas, 4) ./ W;
        p0 = y - f.*x;
        p0(~isfinite(p0)) = 0;
    end

end
