function [x] = tkd(f, mask, vsz, bdir, thr)
%TKD Truncated k-space division.
%
%   [x] = TKD(f, mask, vsz, [bdir], [thr]);
%
%   Inputs
%   ------
%       f       unwrapped local field/phase (3d/4d array).
%       mask    binary mask of region of interest (3d array).
%       vsz     voxel size.
%
%       bdir    unit vector of B field direction.
%               default: [0, 0, 1]
%       thr     threshold for k-space filter.
%               default: 0.15
%
%   Outputs
%   -------
%       x       susceptibility map.
%
%   References
%   ----------
%       [1] Shmueli K, de Zwart JA, van Gelderen P, Li TQ, Dodd SJ, Duyn JH.
%       Magnetic susceptibility mapping of brain tissue in vivo using MRI phase
%       data. Magnetic Resonance in Medicine: An Official Journal of the
%       International Society for Magnetic Resonance in Medicine. 2009
%       Dec;62(6):1510-22.

    narginchk(3, 5);

    if nargin < 5, thr = 0.15; end
    if nargin < 4 || isempty(bdir), bdir = [0, 0, 1]; end


    % output
    x = zeros(size(f), 'like', f);

    % prepare inverse dipole kernel
    D = dipoleKernel(size(mask), vsz, bdir, 'k', class(f));

    I = abs(D) <= thr;

    iD = 1 ./ D;
    iD(I) = sign(D(I)) ./ thr;

    f = fft3(f);

    % 4d multi-echo loop
    for t = 1:size(f, 4)
        x(:,:,:,t) = mask .* real(ifft3(iD .* f(:,:,:,t)));
    end

end
