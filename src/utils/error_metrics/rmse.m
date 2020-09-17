function [e] = rmse(x, y, mask)
%RMSE [e] = rmse(x, y, mask)

    narginchk(2, 3);

    if nargin < 3
        mask = true(size(x));
    end

    e = sqrt(mse(x, y, mask));

end
