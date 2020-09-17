function [w] = dipoleKspaceWeightsIlsqr(D, n, pa, pb)
%DIPOLEKSPACEWEIGHTSILSQR
%   Equation (10) in [1].
%
%   [w] = DIPOLEKSPACEWEIGHTSILSQR(D, n, pa, pb);
%
%   References
%   ----------
%       [1] Li W, Wang N, Yu F, Han H, Cao W, Romero R, Tantiwongkosi B, Duong
%       TQ, Liu C. A method for estimating and removing streaking artifacts in
%       quantitative susceptibility mapping. Neuroimage. 2015 Mar 1;108:111-22.

    narginchk(1, 4);

    if nargin < 4, pb = 30; end
    if nargin < 3 || isempty(pa), pa = 1; end
    if nargin < 2 || isempty(n), n = 0.001; end

    w = abs(D).^n;

    ab = prctile(vec(w), [pa, pb]);

    w = (w - ab(1)) ./ (ab(2) - ab(1));
    w(w < 0) = 0;
    w(w > 1) = 1;

end
