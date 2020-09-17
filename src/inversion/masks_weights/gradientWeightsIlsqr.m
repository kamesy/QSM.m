function [wx, wy, wz] = gradientWeightsIlsqr(x, mask, vsz, pmin, pmax)
%GRADIENTWEIGHTSILSQR
%   Equation (15) in [1].
%
%   [wx, wy, wz] = GRADIENTWEIGHTSILSQR(x, mask, vsz, pmin, pmax);
%
%   References
%   ----------
%       [1] Li W, Wang N, Yu F, Han H, Cao W, Romero R, Tantiwongkosi B, Duong
%       TQ, Liu C. A method for estimating and removing streaking artifacts in
%       quantitative susceptibility mapping. Neuroimage. 2015 Mar 1;108:111-22.

    narginchk(1, 5);

    if nargin < 5, pmax = 70; end
    if nargin < 4 || isempty(pmin), pmin = 50; end
    if nargin < 3 || isempty(vsz), vsz = [1, 1, 1]; end
    if nargin < 2 || isempty(mask), mask = true(size(x(:,:,:,1))); end


    if ndims(x) == 3
        [wx, wy, wz] = gradientWeights_(x, mask, vsz, pmin, pmax);

    elseif ndims(x) == 4
        for t = size(x, 4):-1:1
            [wx(:,:,:,t), wy(:,:,:,t), wz(:,:,:,t)] = ...
                gradientWeights_(x(:,:,:,t), mask, vsz, pmin, pmax);
        end
    end

end


function [wx, wy, wz] = gradientWeights_(x, mask, vsz, pmin, pmax)

    [wx, wy, wz] = gradf(x, mask, vsz);

    wx = gradientWeights__(wx, mask, pmin, pmax);
    wy = gradientWeights__(wy, mask, pmin, pmax);
    wz = gradientWeights__(wz, mask, pmin, pmax);

end


function [w] = gradientWeights__(w, mask, pmin, pmax)

    thr = prctile(w(mask), [pmin, pmax]);

    I1 = w < thr(1);
    I0 = w > thr(2);
    I = ~I0 & ~I1;

    w(I1) = 1;
    w(I) = (thr(2) - w(I)) / (thr(2) - thr(1));
    w(I0) = 0;

    w = mask .* w;

end
