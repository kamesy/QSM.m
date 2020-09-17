function [mem] = getFreeMemory(opt)
%GETFREEMEMORY [mem] = getFreeMemory(opt)

    if nargin < 1, opt = '-m'; end

    mem = 0;

    if isunix
        [s, m] = unix(['free ', opt, ' | grep Mem']);
        if s ~= 0
            warning('%s', m)
        else
            mem = str2double(regexp(m, '[0-9]*', 'match'));
        end
    else
        uv = memory;
        mem = uv.MemAvailableAllArrays;
    end

end
