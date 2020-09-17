function [dx, dy, dz] = gradb(u, mask, vsz)
%GRADB First order backward difference gradient.
%   Boundary condition: u^{k+1} - 2u^{k} + u^{k-1} = 0.
%
%   [dx, dy, dz] = GRADB(u, [mask], [vsz]);
%
%   See also GRADF, GRADC, GRADBADJ

    narginchk(1, 3);

    if nargin < 3,  vsz = [1, 1, 1]; end
    if nargin < 2 || isempty(mask), mask = true(size(u)); end

    validateinputs(u, mask, vsz);

    mask = logical(mask);
    vsz = double(vsz);

    [dx, dy, dz] = grad_(u, mask, vsz);

end


function [dx, dy, dz] = grad_(u, mask, vsz)

    dx = zeros(size(u), 'like', u);
    dy = zeros(size(u), 'like', u);
    dz = zeros(size(u), 'like', u);

    if all(vec(mask))
        gradb_mex(dx, dy, dz, u, vsz);
    else
        gradbm_mex(dx, dy, dz, u, mask, vsz);
    end

end


%{
function [dx, dy, dz] = grad_(u, h)

    ih = 1 ./ h;

    dx = u - circshift(u, [1,0,0]);
    dx(1,:,:) = dx(2,:,:);
    dx = ih(1) .* dx;

    dy = u - circshift(u, [0,1,0]);
    dy(:,1,:) = dy(:,2,:);
    dy = ih(2) .* dy;

    dz = u - circshift(u, [0,0,1]);
    dz(:,:,1) = dz(:,:,2);
    dz = ih(3) .* dz;

end
%}


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
