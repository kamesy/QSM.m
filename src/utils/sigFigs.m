function [x] = sigFigs(x, n)
%SIGFIGS [x] = sigFigs(x, n)

    %x = str2num(num2str(x, n)); %#ok<ST2NM>
    x = round(x, n, 'significant');

end
