function [h] = smvKernel(sz, r, vsz, T)
%SMVKERNEL [h] = smvKernel(sz, r, [vsz], [T])

    narginchk(2, 4);

    if nargin < 4, T = 'double'; end
    if nargin < 3 || isempty(vsz), vsz = [1, 1, 1]; end

    if isscalar(vsz)
        vsz = [vsz, vsz, vsz];
    end


    low  = floor(sz / 2);
    high = low - ~mod(sz, 2);

    low = vsz .* low;
    high = vsz .* high;

    [X, Y, Z] = ndgrid(-low(1) : vsz(1) : high(1), ...
                       -low(2) : vsz(2) : high(2), ...
                       -low(3) : vsz(3) : high(3));

    h = (Y.*Y + X.*X + Z.*Z) <= r^2;

    h = h ./ sum(vec(h));
    h = cast(h, T);

end
