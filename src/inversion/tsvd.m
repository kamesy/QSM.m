function [x] = tsvd(f, mask, vsz, bdir, thr)
%TSVD Truncated singular value decomposition.
%
%   [x] = TSVD(f, mask, vsz, [bdir], [thr]);
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
%       [1] Wharton S, Schäfer A, Bowtell R. Susceptibility mapping in the
%       human brain using threshold‐based k‐space division. Magnetic resonance
%       in medicine. 2010 May;63(5):1292-304.

    narginchk(3, 5);

    if nargin < 5, thr = 0.15; end
    if nargin < 4 || isempty(bdir), bdir = [0, 0, 1]; end


    % output
    x = zeros(size(f), 'like', f);

    % prepare inverse dipole kernel
    D = dipoleKernel(size(mask), vsz, bdir, 'k', class(f));

    I = abs(D) <= thr;

    iD = 1 ./ D;
    iD(I) = 0;

    f = fft3(f);

    % 4d multi-echo loop
    for t = 1:size(f, 4)
        x(:,:,:,t) = mask .* real(ifft3(iD .* f(:,:,:,t)));
    end

end
