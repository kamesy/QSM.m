function [y] = fitPoly2d(x, n, mask, vsz)
%FITPOLY2D Fit 2d polynomial of order n to 3d multi-echo data
%
%   [y] = FITPOLY2D(x, n, [mask], [vsz]);
%
%   Inputs
%   ------
%       x       multi-echo data, 3d/4d array.
%       n       order of polynomial to fit.
%
%       mask    binary mask of voxels to include in fit.
%               default = ones(size(x))
%       vsz     voxel size.
%               default = [1, 1, 1]
%
%   Outputs
%   -------
%       y       polynomial fit of x
%
%   Notes
%   -----
%       Adapted from John D'Errico's polyfitn toolbox:
%       https://www.mathworks.com/matlabcentral/fileexchange/34765-polyfitn
%
%   See also POLYFITN, FITPOLY3D

    narginchk(2, 4);

    if nargin < 4, vsz = [1, 1, 1]; end
    if nargin < 3 || isempty(mask), mask = true(size(x(:,:,:,1))); end

    validateinputs(x, n, mask, vsz);


    T = class(x);
    sz = size(mask);
    sz = cast(sz, T);
    vsz = cast(vsz, T);
    mask = logical(mask);

    % build the model
    k = buildmodel(n, 2);
    k = cast(k, T);

    % generate grid
    low = floor(sz / 2);
    high = low - ~mod(sz, 2);
    low = vsz .* low;
    high = vsz .* high;
    [X, Y] = ndgrid(-low(1):vsz(1):high(1), -low(2):vsz(2):high(2));

    I = [vec(X), vec(Y)];

    % build the design matrix
    A = ones(size(I, 1), size(k, 1), T);
    for ii = 1:size(A, 2)
        A(:,ii) = I(:,1).^k(ii, 1) .* I(:,2).^k(ii, 2);
    end

    y = zeros(size(x), T);

    for slice = 1:sz(3)
        m = mask(:,:,slice);

        % scale to unit variance
        I = [X(m), Y(m)];
        s = sqrt(diag(cov(I)));
        s(s == 0) = 1;
        I = I * diag(1 ./ s);

        % build the design matrix
        Am = ones(size(I, 1), size(k, 1), T);
        ss = ones(1, size(Am, 2), T);   % scaling factor
        for ii = 1:size(Am, 2)
            Am(:,ii) = I(:,1).^k(ii, 1) .* I(:,2).^k(ii, 2);
            ss(ii) = s(1)^k(ii, 1) * s(2)^k(ii, 2);
        end

        [Q, R, E] = qr(Am, 0);

        for t = size(x, 4):-1:1
            x_ = x(:,:,slice,t);
            xm = x_(m);

            c(E) = R \ (Q' * xm);
            c = c ./ ss;    % apply scaling
            y_ = A * vec(c);

            y(:,:,slice,t) = reshape(y_, sz(1:2));
        end
    end

    y(~isfinite(y)) = 0;

end


function [m] = buildmodel(order, p)

    if p == 0
        m = [];

    elseif order == 0
        m = zeros(1, p);

    elseif p == 1
        m = (order:-1:0).';

    else
        m = zeros(0, p);
        for k = order:-1:0
            t = buildmodel(order-k, p-1);
            m = [m; [repmat(k, size(t, 1), 1), t]]; %#ok<AGROW>
        end
    end

end



function [] = validateinputs(x, n, mask, vsz)

    sz = size(x);

    if ndims(x) < 4
        nd = 3;
    else
        nd = 4;
    end

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', nd, 'finite'};
    validateattributes(x, classes, attributes, mfilename, 'x', 1);

    classes = {'numeric'};
    attributes = {'real', 'scalar', 'finite', 'integer', 'nonnegative'};
    validateattributes(n, classes, attributes, mfilename, 'n', 2);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'size', sz(1:3), 'finite', 'binary'};
    validateattributes(mask, classes, attributes, mfilename, 'mask', 3);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 4);

end
