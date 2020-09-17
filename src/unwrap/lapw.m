function [d2phas] = lapw(phas, vsz)
%LAPW Discrete Laplacian of wrapped phase data.
%
%   [d2phas] = LAPW(phas, vsz);
%
%   Inputs
%   ------
%       phas    wrapped phase (3d array).
%       vsz     voxel size (3 element vector).
%
%   Outputs
%   -------
%       d2phas  Laplacian of unwrapped phase.
%
%   Notes
%   -----
%       The Laplacian is computed using second order central finite differences
%       on the complex phase:
%
%           d2u/dx2 = arg(exp( i * [u(x-1) - 2u(x) + u(x+1)] ))

    narginchk(2, 2);

    validateinputs(phas, vsz);
    vsz = double(vsz);

    d2phas = lapw_(phas, vsz);

end


function [d2u] = lapw_(u, h)

    d2u = zeros(size(u), 'like', u);
    lapw_mex(d2u, u, h);

end


%{
function [d2u] = lapw_(u, h)

    ih2 = 1./(h.*h);

    % atan2(sin, cos) is faster than angle(exp(1i))
    tmp = circshift(u, [-1,0,0]) - u;
    tmp(end,:,:) = 0;
    d2u = ih2(1).*atan2(sin(tmp), cos(tmp));

    tmp = u - circshift(u, [1,0,0]);
    tmp(1,:,:) = 0;
    d2u = d2u - ih2(1).*atan2(sin(tmp), cos(tmp));

    tmp = circshift(u, [0,-1,0]) - u;
    tmp(:,end,:) = 0;
    d2u = d2u + ih2(2).*atan2(sin(tmp), cos(tmp));

    tmp = u - circshift(u, [0,1,0]);
    tmp(:,1,:) = 0;
    d2u = d2u - ih2(2).*atan2(sin(tmp), cos(tmp));

    tmp = circshift(u, [0,0,-1]) - u;
    tmp(:,:,end) = 0;
    d2u = d2u + ih2(3).*atan2(sin(tmp), cos(tmp));

    tmp = u - circshift(u, [0,0,1]);
    tmp(:,:,1) = 0;
    d2u = d2u - ih2(3).*atan2(sin(tmp), cos(tmp));

end
%}


function [] = validateinputs(phas, vsz)

    classes = {'single', 'double'};
    attributes = {'real', 'ndims', 3, 'finite'};
    validateattributes(phas, classes, attributes, mfilename, 'phas', 1);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 2);

end
