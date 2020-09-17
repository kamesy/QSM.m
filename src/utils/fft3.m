function [X] = fft3(x)
%FFT3 Summary of this function goes here
%   Detailed explanation goes here

    X = fft(fft(fft(x, [], 3), [], 2), [], 1);

end
