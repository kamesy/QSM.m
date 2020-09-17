function [w] = laplacianWeightsIlsqr(f, mask, vsz, pmin, pmax)
%LAPLACIANWEIGHTSILSQR
%   Equation (7) in [1].
%
%   [w] = LAPLACIANWEIGHTSILSQR(f, mask, vsz, pmin, pmax);
%
%   References
%   ----------
%       [1] Li W, Wang N, Yu F, Han H, Cao W, Romero R, Tantiwongkosi B, Duong
%       TQ, Liu C. A method for estimating and removing streaking artifacts in
%       quantitative susceptibility mapping. Neuroimage. 2015 Mar 1;108:111-22.

    narginchk(1, 5);

    if nargin < 5, pmax = 99.9; end
    if nargin < 4 || isempty(pmin), pmin = 60; end
    if nargin < 3 || isempty(vsz), vsz = [1, 1, 1]; end
    if nargin < 2 || isempty(mask), mask = true(size(f(:,:,:,1))); end


    if ndims(f) == 3
        w = laplacianWeights_(f, mask, vsz, pmin, pmax);

    elseif ndims(f) == 4
        for t = size(f, 4):-1:1
            w(:,:,:,t) = ...
                laplacianWeights_(f(:,:,:,t), mask, vsz, pmin, pmax);
        end
    end

end


function [w] = laplacianWeights_(f, mask, vsz, pmin, pmax)

    w = zeros(size(f), 'like', f);

    d2f = lap(f, mask, vsz);

    thr = prctile(d2f(mask), [pmin, pmax]);

    I = thr(1) <= d2f & d2f <= thr(2);

    w(d2f < thr(1)) = 1;
    w(I) = (thr(2) - d2f(I)) ./ (thr(2) - thr(1));
    w(d2f > thr(2)) = 0;

end
