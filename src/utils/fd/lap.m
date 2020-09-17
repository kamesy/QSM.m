function [d2u] = lap(u, mask, vsz)
%LAP Second order central difference Laplacian.
%
%   [d2u] = LAP(u, [mask], [vsz]);

    narginchk(1, 3);

    if nargin < 3,  vsz = [1, 1, 1]; end
    if nargin < 2 || isempty(mask), mask = true(size(u(:,:,:,1))); end

    validateinputs(u, mask, vsz);

    mask = logical(mask);
    vsz = double(vsz);

    d2u = lap_(u, mask, vsz);

end


function [d2u] = lap_(u, mask, vsz)

    d2u = zeros(size(u), 'like', u);
    lap1_mex(d2u, u, mask, vsz);

end


function [] = validateinputs(u, mask, vsz)

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', 3, 'finite'};
    validateattributes(u, classes, attributes, mfilename, 'u', 1);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'size', size(u), 'finite', 'binary'};
    validateattributes(mask, classes, attributes, mfilename, 'mask', 2);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 3);

end
