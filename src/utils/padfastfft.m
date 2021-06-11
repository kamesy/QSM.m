function [u] = padfastfft(u, padsize, method, direction)
%PADFASTFFT [u] = padfastfft(u, padsize, [padval|method], [direction])
%   See also UNPADFASTFFT, NEXTFASTFFT, PADARRAY

    narginchk(1, 4);

    if nargin < 2, padsize = zeros(1, ndims(u)); end
    if nargin < 3, method = 0; end
    if nargin < 4
        direction = 'both';
    else
        direction = validatestring(direction, {'pre', 'post', 'both'});
    end


    sz = size(u);
    padsize = reshape(padsize, 1, []);

    if length(padsize) < length(sz)
        padsize(1, length(sz)) = 0;

    elseif length(padsize) > length(sz)
        tmp = sz;
        sz = ones(1, length(padsize));
        sz(1:length(tmp)) = tmp;
    end


    if ~strcmpi(direction, 'both')
        pad = nextfastfft(sz + padsize) - sz;
        pad(padsize == 0) = 0;
        u = padarray(u, pad, method, direction);

    else
        pad = nextfastfft(sz + 2*padsize) - sz;
        pad(padsize == 0) = 0;

        oddsz = bitand(sz, 1);
        oddpad = bitand(pad, 1);

        oddpre = bitand(oddpad, oddsz);
        oddpost = bitand(oddpad, ~oddsz);

        pad	= floor(pad / 2);

        if any(oddpad)
            u = padarray(u, pad + oddpre, method, 'pre');
            u = padarray(u, pad + oddpost, method, 'post');
        else
            u = padarray(u, pad, method, 'both');
        end
    end

end
