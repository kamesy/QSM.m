function [x] = ifft3(X)
%IFFT3 Summary of this function goes here
%   Detailed explanation goes here

    x = ifft(ifft(ifft(X, [], 3), [], 2), [], 1);

end
