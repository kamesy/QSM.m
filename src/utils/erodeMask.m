function [mask] = erodeMask(mask, vx)
%ERODEMASK Erode 3d binary mask.
%
%   [mask] = ERODEMASK(mask, vx);
%
%   Inputs
%   ------
%       mask    3d binary mask.
%       vx      number of voxels to erode.
%
%   Outputs
%   -------
%       mask    eroded binary mask.
%
%   See also DILATEMASK

    narginchk(2, 2);
    validateinputs(mask, vx);

    if vx < 1
        return;
    end

    T = class(mask);
    mask = logical(mask);
    mask = erodeMask_(mask, vx);
    mask = cast(mask, T);

end


function [mask] = erodeMask_(mask, vx)

    try
        % fast implementation in C: ./poisson_solver/mex/boundary_mask_l1_mex.c
        b = false(size(mask));
        boundary_mask_l1_mex(b, mask, vx);
        mask = mask - b;

    catch ex
        warning(ex.identifier, '%s\n', ex.message, 'using fallback method');
        for ii = 1:vx
            mask = mask - bwperim(mask, 6);
        end
    end

end


function [] = validateinputs(mask, vx)

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'finite', 'binary'};
    validateattributes(mask, classes, attributes, mfilename, 'mask', 1);

    classes = {'numeric'};
    attributes = {'real', 'scalar', 'finite', 'nonnegative', 'integer'};
    validateattributes(vx, classes, attributes, mfilename, 'vx', 2);

end
