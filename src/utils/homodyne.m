function [u] = homodyne(u, n)
%HOMODYNE [u] = homodyne(u, n)

    narginchk(2, 2)

    if length(n) == 1, n = [n, n]; end
    if n(1) > size(u, 1), n = [size(u, 1), n(2)]; end
    if n(2) > size(u, 2), n = [n(1), size(u, 2)]; end


    w = hann2d(size(u), n);

    for t = size(u, 4):-1:1
        for k = size(u, 3):-1:1
            slice = u(:,:,k,t);
            slice = slice .* conj(ifft2(w .* fft2(slice)));
            u(:,:,k,t) = slice;
        end
    end

end


function [w] = hann2d(sz, n)

    s1 = sz(1);
    s2 = sz(2);

    n1 = n(1);
    n2 = n(2);

    x = circshift((0:n1-1)-floor((n1-1)/2), [0, -floor((n1-1)/2)]);
    y = circshift((0:n2-1)-floor((n2-1)/2), [0, -floor((n2-1)/2)]);
    x = .5 * (1 + cos((2*pi/n1) * x));
    y = .5 * (1 + cos((2*pi/n2) * y));

    kx = zeros(1, s1);
    ky = zeros(1, s2);

    kx(1:ceil(n1/2)) = x(1:ceil(n1/2));
    kx(s1-n1+(ceil(n1/2)+1):end) = x(ceil(n1/2)+1:end);

    ky(1:ceil(n2/2)) = y(1:ceil(n2/2));
    ky(s2-n2+(ceil(n2/2)+1):end) = y(ceil(n2/2)+1:end);

    w = kx'*ky;

end
