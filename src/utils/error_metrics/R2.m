function [e] = R2(x, y, mask)
%R2 [e] = R2(x, y, mask)

    narginchk(2, 3);

    if nargin < 3
        x = vec(x);
        y = vec(y);
    else
        x = x(mask);
        y = y(mask);
    end

    e = 1 - sum((x - y).^2) / sum((y - mean(y)).^2);

end
