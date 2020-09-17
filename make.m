function [] = make(openmp)

    if nargin < 1
        openmp = 1;
    end


    pwd_ = pwd();
    mdir = fileparts(mfilename('fullpath'));

    % doesn't work on matlab versions < 2016b
    makefiles = dir('src/**/make.m');

    if isempty(makefiles)
        makefiles = [
            struct('folder', fullfile(mdir, 'src/unwrap/mex')), ...
            struct('folder', fullfile(mdir, 'src/inversion/mex')), ...
            struct('folder', fullfile(mdir, 'src/utils/fd/mex')), ...
            struct('folder', fullfile(mdir, 'src/utils/poisson_solver/mex')), ...
        ];
    end

    ex = [];
    try
        for ii = 1:length(makefiles)
            cd(makefiles(ii).folder);
            make(openmp);
        end
    catch ex
        % void
    end

    cd(pwd_);

    if ~isempty(ex)
        rethrow(ex);
    end

end
