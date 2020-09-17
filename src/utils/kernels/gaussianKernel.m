function [h] = gaussianKernel(sz, sigma, vsz, center, T)
%GAUSSKERNEL [h] = gaussianKernel(sz, sigma, [vsz], [center], [T])

    narginchk(2, 5);

    if nargin < 5, T = 'double'; end
    if nargin < 4 || isempty(center), center = [0, 0, 0]; end
    if nargin < 3 || isempty(vsz), vsz = [1, 1, 1]; end

    if isscalar(sigma)
        sigma = [sigma, sigma, sigma];
    end


    n = (sz - 1) / 2;

    [X, Y, Z] = ndgrid((-n(1):n(1)) + center(1), ...
                       (-n(2):n(2)) + center(2), ...
                       (-n(3):n(3)) + center(3));

    X = X .* vsz(1);
    Y = Y .* vsz(2);
    Z = Z .* vsz(3);

    sigma2 = sigma .* sigma;

    h = (X.*X)/sigma2(1) + (Y.*Y)/sigma2(2) + (Z.*Z)/sigma2(3);
    h = exp(-h / 2);

    h(h < eps(max(abs(vec(h))))) = 0;

    n = sum(vec(h));

    if n ~= 0
        h = h ./ n;
    end

    h = cast(h, T);

end
