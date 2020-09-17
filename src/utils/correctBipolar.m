function [phas] = correctBipolar(phas, mask, vsz)
%CORRECTBIPOLAR Phase correction for bipolar readout gradients.
%
%   [phas] = CORRECTBIPOLAR(phas, mask, vsz);
%
%   Inputs
%   ------
%       phas    4d multi-echo phase data. Minimum 3 echoes needed.
%       mask    binary mask for polynomial fit.
%       vsz     voxel size for polynomial fit.
%
%   Outputs
%   -------
%       phas    corrected phase.
%
%   Notes
%   -----
%       This method will not work on highly wrapped phase data, ie. when the
%       jump from first to third echo contains wraps. If `dT` contains wraps,
%       unwrap phase first before applying correction.
%
%   References
%   ----------
%       [1] Li J, Chang S, Liu T, Jiang H, Dong F, Pei M, Wang Q, Wang Y.
%       Phase-corrected bipolar gradients in multi-echo gradient-echo sequences
%       for quantitative susceptibility mapping. Magnetic Resonance Materials
%       in Physics, Biology and Medicine. 2015 Aug 1;28(4):347-55.
%
%   See also FITPOLY3D

    narginchk(3, 3);

    if size(phas, 4) < 3
        warning('minimum three echoes needed for correction');
        return;
    end

    dT = phas(:,:,:,2) - 0.5*(phas(:,:,:,1) + phas(:,:,:,3));
    dT = fitPoly3d(dT, 1, mask, vsz);
    dT = mask .* dT;

    for t = 2:2:size(phas, 4)
        phas(:,:,:,t) = phas(:,:,:,t) - dT;
    end

end
