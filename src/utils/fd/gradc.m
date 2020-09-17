function [dx, dy, dz] = gradc(u, mask, vsz)
%GRADC First order centered difference gradient.
%   Boundary condition: u^{k+1} - 2u^{k} + u^{k-1} = 0.
%
%   [dx, dy, dz] = GRADC(u, [mask], [vsz]);
%
%   See also GRADF, GRADB

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
        gradc_mex(dx, dy, dz, u, vsz);
    else
        gradcm_mex(dx, dy, dz, u, mask, vsz);
    end

end


%{
function [dx, dy, dz] = grad_(u, h)

    ih = 1 ./ h;

    dx = circshift(u, [-1,0,0]) - circshift(u, [1,0,0]);
    dx = 0.5 .* ih(1) .* dx;
    dx(1,:,:) = ih(1) .* (u(2,:,:) - u(1,:,:));
    dx(end,:,:) = ih(1) .* (u(end,:,:) - u(end-1,:,:));

    dy = circshift(u, [0,-1,0]) - circshift(u, [0,1,0]);
    dy = 0.5 .* ih(2) .* dy;
    dy(:,1,:) = ih(2) .* (u(:,2,:) - u(:,1,:));
    dy(:,end,:) = ih(2) .* (u(:,end,:) - u(:,end-1,:));

    dz = circshift(u, [0,0,-1]) - circshift(u, [0,0,1]);
    dz = 0.5 .* ih(3) .* dz;
    dz(:,:,1) = ih(3) .* (u(:,:,2) - u(:,:,1));
    dz(:,:,end) = ih(3) .* (u(:,:,end) - u(:,:,end-1));

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
