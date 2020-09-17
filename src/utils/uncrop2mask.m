function [u, ix, iy, iz] = uncrop2mask(x, mask)
%UNCROP2MASK [u, ix, iy, iz] = uncrop2mask(x, mask)
%   See also CROP2MASK, CROPINDICES

    narginchk(2, 2);

    u = zeros(size(mask), 'like', x);
    [ix, iy, iz] = cropIndices(mask);
    u(ix,iy,iz) = x;

end
