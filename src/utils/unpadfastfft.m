function [u] = unpadfastfft(u, sz0, padsize, direction)
%UNPADFASTFFT [u] = unpadfastfft(u, sz0, padsize, [direction])
%   See also PADFASTFFT, NEXTFASTFFT, PADARRAY

    narginchk(2, 4)

    if nargin < 3, padsize = zeros(1, length(sz0)); end
    if nargin < 4
        direction = 'both';
    else
        direction = validatestring(direction, {'pre', 'post', 'both'});
    end


    sz = size(u);
    sz0 = reshape(sz0, 1, []);
    padsize = reshape(padsize, 1, []);

    if length(padsize) < length(sz)
        padsize(length(sz)) = 0;

    elseif length(padsize) > length(sz)
        tmp = sz;
        sz = ones(1, length(padsize));
        sz(1:length(tmp)) = tmp;
    end


    if length(sz0) ~= length(sz)
        error('dimension mismatch')
    end

    if ~all(sz >= sz0)
        error('unpad size must be smaller or equal to current size')
    end


    R = cell(1, length(sz));
    R(:) = {':'};

    if strcmpi(direction, 'pre')
        for d = 1:length(sz)
            s0 = sz0(d);
            s1 = sz(d);
            R{d} = s1-s0+1:s1;
        end

    elseif strcmpi(direction, 'post')
        for d = 1:length(sz)
            R{d} = 1:sz(d);
        end

    else
        for d = 1:length(sz)
            s0 = sz0(d);
            s1 = sz(d);

            if padsize(d) == 0
                pad = 0;
            else
                pad = nextfastfft(s0 + 2*padsize(d)) - s0;
            end

            odds0 = bitand(s0, 1);
            oddpad = bitand(pad, 1);

            oddpre = bitand(oddpad, odds0);
            oddpost = bitand(oddpad, ~odds0);

            pad	= floor(pad / 2);

            if any(oddpad)
                R{d} = (pad+oddpre)+1:s1-(pad+oddpost);
            else
                R{d} = pad+1:s1-pad;
            end
        end
    end

    u = u(R{:});

end
