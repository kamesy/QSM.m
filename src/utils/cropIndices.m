function [ix, iy, iz] = cropIndices(x)
%CROPINDICES [ix, iy, iz] = cropIndices(x)
%   See also CROP2MASK, UNCROP2MASK

    narginchk(1, 1);

    s1 = sum(x, 1);
    s2 = sum(x, 2);

    ix = find(sum(s2, 3));
    iy = find(sum(s1, 3));
    iz = find(sum(s1, 2));

    ix = vec(ix);
    iy = vec(iy);
    iz = vec(iz);

end
