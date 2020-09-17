function [L] = laplaceKernel(vsz, T)
%LAPKERNEL [L] = laplaceKernel([vsz], [T])

    narginchk(0, 2);

    if nargin < 2, T = 'double'; end
    if nargin < 1 || isempty(vsz), vsz = [1, 1, 1]; end

    if isscalar(vsz)
        vsz = [vsz, vsz, vsz];
    end


    ihx2 = 1 ./ vsz(1)^2;
    ihy2 = 1 ./ vsz(2)^2;
    ihz2 = 1 ./ vsz(3)^2;
    ihh2 = -2 * (ihx2 + ihy2 + ihz2);

    % Laplace Kernel
    L(:,:,1) = [0   0  0;   0  ihz2  0  ; 0   0  0];
    L(:,:,2) = [0 ihx2 0; ihy2 ihh2 ihy2; 0 ihx2 0];
    L(:,:,3) = [0   0  0;   0  ihz2  0  ; 0   0  0];

    L = cast(L, T);

end
