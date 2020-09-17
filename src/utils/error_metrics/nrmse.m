function [e] = nrmse(x, y, mask)
%NRMSE [e] = nrmse(x, y, mask)

    narginchk(2, 3);

    if nargin < 3
        mask = true(size(x));
    end

    e = rmse(x, y, mask);
    e = e / range(vec(x));

end
