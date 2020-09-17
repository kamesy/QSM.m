function [du] = gradbAdj(x, y, z, mask, vsz)
%GRADBADJ Adjoint of first order backward difference gradient.
%
%   [du] = GRADBADJ(x, y, z, [mask], [vsz]);
%
%   See also GRADB

    narginchk(3, 5);

    if nargin < 5,  vsz = [1, 1, 1]; end
    if nargin < 4 || isempty(mask), mask = true(size(x(:,:,:,1))); end

    validateinputs(x, y, z, mask, vsz);

    if isa(x, 'single') || isa(y, 'single') || isa(z, 'single')
        x = cast(x, 'single');
        y = cast(y, 'single');
        z = cast(z, 'single');
    end

    mask = logical(mask);
    vsz = double(vsz);

    du = gradAdj_(x, y, z, mask, vsz);

end


function [du] = gradAdj_(x, y, z, mask, vsz)

    du = zeros(size(x), 'like', x);

    if all(vec(mask))
        gradb_adj_mex(du, x, y, z, vsz);
    else
        gradbm_adj_mex(du, x, y, z, mask, vsz);
    end

end


%{
function [du] = gradAdj_(x, y, z, h)

    ih = -1 ./ h;

    d = circshift(x, [-1,0,0]) - x;
    d(end,:,:) = d(end-1,:,:);
    d = ih(1) .* d;

    du = d;

    d = circshift(y, [0,-1,0]) - y;
    d(:,end,:) = d(:,end-1,:);
    d = ih(2) .* d;

    du = du + d;

    d = circshift(z, [0,0,-1]) - z;
    d(:,:,end) = d(:,:,end-1);
    d = ih(3) .* d;

    du = du + d;

end
%}


function [] = validateinputs(x, y, z, mask, vsz)

    sz = size(x);

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', 3, 'finite'};
    validateattributes(x, classes, attributes, mfilename, 'x', 1);

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', 3, 'size', sz, 'finite'};
    validateattributes(y, classes, attributes, mfilename, 'y', 2);

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', 3, 'size', sz, 'finite'};
    validateattributes(z, classes, attributes, mfilename, 'z', 3);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'size', sz, 'finite', 'binary'};
    validateattributes(mask, classes, attributes, mfilename, 'mask', 4);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 5);

end
