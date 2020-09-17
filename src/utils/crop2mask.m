function [u, ix, iy, iz] = crop2mask(x, mask)
%CROP2MASK [u, ix, iy, iz] = crop2mask(x, [mask])
%   See also UNCROP2MASK, CROPINDICES

    narginchk(1, 2);
    if nargin < 2, mask = x; end

    [ix, iy, iz] = cropIndices(mask);
    u = x(ix,iy,iz);

end
