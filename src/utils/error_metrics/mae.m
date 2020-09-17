function [e] = mae(x, y, mask)
%MAE [e] = mae(x, y, mask)

    narginchk(2, 3)

    if nargin < 3
        mask = true(size(x));
    end

    if ~isa(mask, 'logical')
        mask = logical(mask);
    end

    x = vec(mask .* (x - y));

    e = sum(abs(x));
    e = e / sum(vec(mask));

end
