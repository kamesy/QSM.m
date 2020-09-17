function [mx, my, mz] = gradientMaskMedi(mag, mask, vsz, p)
%GRADIENTMASKMEDI Binary mask for gradient weighting.
%   Equation (4) in [1].
%
%   [mx, my, mz] = GRADIENTMASKMEDI(mag, mask, vsz, p);
%
%   References
%   ----------
%       [1] Liu J, Liu T, de Rochefort L, Ledoux J, Khalidov I, Chen W,
%       Tsiouris AJ, Wisnieff C, Spincemaille P, Prince MR, Wang Y. Morphology
%       enabled dipole inversion for quantitative susceptibility mapping using
%       structural consistency between the magnitude image and the
%       susceptibility map. Neuroimage. 2012 Feb 1;59(3):2560-8.

    narginchk(1, 4);

    if nargin < 4, p = 30; end
    if nargin < 3 || isempty(vsz), vsz = [1, 1, 1]; end
    if nargin < 2 || isempty(mask), mask = true(size(mag(:,:,:,1))); end


    if ndims(mag) == 3
        [mx, my, mz] = gradientMask_(mag, mask, vsz, p);

    elseif ndims(mag) == 4
        for t = size(mag, 4):-1:1
            [mx(:,:,:,t), my(:,:,:,t), mz(:,:,:,t)] = ...
                gradientMask_(mag(:,:,:,t), mask, vsz, p);
        end
    end

end


function [mx, my, mz] = gradientMask_(mag, mask, vsz, p)

    mag = mask .* mag ./ max(abs(vec(mag)));
    [mx, my, mz] = gradf(mag, mask, vsz);

    mx = abs(mx);
    my = abs(my);
    mz = abs(mz);

    thr = prctile([mx(mask); my(mask); mz(mask)], 100 - p);

    mx = mx < thr;
    my = my < thr;
    mz = mz < thr;

end
