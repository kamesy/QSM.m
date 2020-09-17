function [ssidx, ssmap] = ssim(x, y, sigma, L, K, abc)
%SSIM [ssidx, ssmap] = ssim(x, y, sigma, L, K, abc)
%
%   References
%   ----------
%       [1] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%       Quality Assessment: From Error Visibility to Structural Similarity",
%       IEEE Transactions on Image Processing, Volume 13, Issue 4, pp. 600-
%       612, 2004.

    narginchk(2, 6)

    if nargin < 3 || isempty(sigma)
        sigma = 1.5;
    end

    if nargin < 4 || isempty(L)
        L = 1;
    end

    if nargin < 5 || isempty(K)
        K = [0.01, 0.03, 0.03];
        C = (K * L).^2;
        C(3) = C(2) / 2;
    else
        C = (K * L).^2;
    end

    if nargin < 6
        abc = [1, 1, 1];
    end


    sz = size(x);
    hsz = ceil(5 * sigma);

    x = padfastfft(x, hsz, 'replicate');
    y = padfastfft(y, hsz, 'replicate');

    h = gaussKernel(size(x), sigma);
    h = fft3(ifftshift(h));

    mux2  = real(ifft3(h .* fft3(x)));                          % Eq [14]
    mux2  = unpadfastfft(mux2, sz, hsz);

    muy2  = real(ifft3(h .* fft3(y)));
    muy2  = unpadfastfft(muy2, sz, hsz);                        % Eq [14]

    muxy  = mux2 .* muy2;
    mux2  = mux2 .* mux2;
    muy2  = muy2 .* muy2;

    sigx2 = real(ifft3(h .* fft3(x .* x)));
    sigx2 = unpadfastfft(sigx2, sz, hsz) - mux2;                % Eq [15]

    sigy2 = real(ifft3(h .* fft3(y .* y)));
    sigy2 = unpadfastfft(sigy2, sz, hsz) - muy2;                % Eq [15]

    sigxy = real(ifft3(h .* fft3(x .* y)));
    sigxy = unpadfastfft(sigxy, sz, hsz) - muxy;                % Eq [16]

    clear x y h

    if (C(2) == 2*C(3)) && all(abc == 1)
        ssmap = ...
            (2*muxy + C(1)) .* (2*sigxy + C(2)) ./ ...
            ((mux2 + muy2 + C(1)) .* (sigx2 + sigy2 + C(2)));   % Eq [13]
    else
        sigxsigy = sqrt(sigx2 .* sigy2);
        l = (2*muxy + C(1)) ./ (mux2 + muy2 + C(1));            % Eq [6]
        c = (2*sigxsigy + C(2)) ./ (sigx2 + sigy2 + C(2));      % Eq [9]
        s = (sigxy + C(3)) ./ (sigxsigy + C(3));                % Eq [10]

        ssmap = l.^abc(1) .* c.^abc(2) .* s.^abc(3);            % Eq [12]
    end

    ssidx = mean(real(vec(ssmap)));

end
