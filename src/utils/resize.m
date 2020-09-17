function [u1] = resize(u, scale, wname, varargin)
%RESIZE [u1] = resize(u, scale, [wname, opt1, opt2])
%   See also WINDOW

    narginchk(2, Inf)

    if nargin < 3, wname = []; end

    if ~isnumeric(scale)
        error('non numeric scaling factors')
    end

    if any(scale <= 0)
        error('scaling factors must be greater than zero: %s', ...
            sprintf('%.2f ', scale))
    end

    if ~isempty(wname) && ~isa(wname, 'function_handle')
        error('wname must be a function_handle, see `help window`')
    end


    if length(scale) == 1
        scale = scale * ones(ndims(u), 1);
    end

    [sz, sz1] = deal(size(u));

    for d = 1:min(length(scale), ndims(u))
        sz1(d) = round(scale(d) * sz(d));
    end


    if isequal(sz, sz1)
        u1 = u;

    elseif ~any(sz1 < sz)
        u1 = resize_(u, sz1, wname, varargin{:});

    else
        u1 = u;
        dims = find(sz1 < sz);
        for d = dims
            sz2 = size(u1);
            R = cell(1, ndims(u1));
            R(:) = {':'};

            inc = floor(sz(d)/sz1(d)) + 1;
            sz2(d) = inc * sz1(d);
            R{d} = 1:inc:sz2(d);

            u1 = resize_(u1, sz2, wname, varargin{:});
            u1 = u1(R{:});
        end

        if ~isequal(size(u1), sz1)
            u1 = resize_(u1, sz1, wname, varargin{:});
        end
    end

end


function [u1] = resize_(u0, sz1, wname, varargin)

    sz0 = size(u0);
    isr = isreal(u0);

    dims = find(sz0 ~= sz1);
    [R0, R1] = resizeIndices(sz0, sz1);

    for d = dims
        u0 = fft(u0, [], d);
    end

    if ~isempty(wname)
        w = getWindow(u0, dims, wname, varargin{:});
        u0 = w .* u0;
    end

    u1 = zeros(sz1, 'like', u0);
    u1(R1{:}) = prod(sz1)/prod(sz0) .* u0(R0{:});

    u1 = processEvenDims(u1, u0);

    for d = dims
        u1 = ifft(u1, [], d);
    end

    if isr
        u1 = real(u1);
    end

end


function [u1] = processEvenDims(u1, u0)

    sz0 = size(u0);
    sz1 = size(u1);
    dims = find(sz0 ~= sz1);
    for d = dims
        s0 = sz0(d);
        s1 = sz1(d);

        if rem(min(s0, s1), 2) == 0
            R  = cell(1, ndims(u1));
            R(:) = {':'};
            R1 = R;

            if s1 > s0      % Upsampling
                R{d}  = ceil((s0+1)/2);
                R1{d} = R{d}+s1-s0;
                u1(R{:})  = 0.5 .* u1(R{:});
                u1(R1{:}) = u1(R{:});

            elseif s1 < s0  % Downsampling
                % TODO?

            end
        end
    end

end


function [w] = getWindow(u0, dims, wname, varargin)

    sz0 = size(u0);
    w = ones(sz0, 'like', u0);
    for d = dims
        w1 = ifftshift(window(wname, sz0(d), varargin{:}));

        szw = ones(1, ndims(u0));
        szw(d) = sz0(d);
        w1 = reshape(w1, szw);

        szw = sz0;
        szw(d) = 1;
        w1 = repmat(w1, szw);

        w = w1 .* w;
    end

end


function [R0, R1] = resizeIndices(sz0, sz1)

    [R0, R1] = deal(cell(1,length(sz0)));
    [R0(:), R1(:)] = deal({':'});

    for d = 1:length(sz0)  %#ok<*AGROW>
        s0 = sz0(d);
        s1 = sz1(d);
        n  = min(s0, s1);

        inds = [1:ceil((n+1)/2), abs(s0-s1) + (ceil((n+1)/2)+1:n)];

        if s1 > s0      % Upsampling
            R1{d} = inds;
        elseif s1 < s0  % Downsampling
            R0{d} = inds;
        end
    end

end
