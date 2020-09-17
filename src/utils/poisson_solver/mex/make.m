function [] = make(openmp, gs_red_black)

    if nargin < 2
        gs_red_black = 1;
    end

    if nargin < 1
        openmp = 1;
    end


    CFLAGS = '$CFLAGS -Wall -Wextra -pedantic -std=c11 -fomit-frame-pointer';
    COPTIMFLAGS = '$COPTIMFLAGS -O2';

    LDFLAGS = '$LDFLAGS';
    LDOPTIMFLAGS = '$LDOPTIMFLAGS -O2';

    LIBS = '-lmwblas';

    if openmp
        CFLAGS = [CFLAGS, ' ', '-fopenmp'];
        LDFLAGS = [LDFLAGS, ' ', '-fopenmp'];
        if isunix && ~ismac
            LIBS = [LIBS, ' ', '-lgomp'];
        end
    else
        CFLAGS = [CFLAGS, ' ', '-Wno-unknown-pragmas -Wno-unused-variable'];
    end


    DEFINES = '';

    if gs_red_black
        DGAUSS_SEIDEL_RED_BLACK = '-DGAUSS_SEIDEL_RED_BLACK';
    else
        DGAUSS_SEIDEL_RED_BLACK = '';
    end

    DEFINES = [DEFINES, ' ', DGAUSS_SEIDEL_RED_BLACK];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % mgpcg_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = { ...
        'mgpcg_mex.c', ...
        'boundary_mask_mex.c', ...
        'coarsen_grid_mex.c', ...
        'correct_mex.c', ...
        'gauss_seidel_mex.c', ...
        'lapmg_mex.c', ...
        'mx_blas.c', ...
        'mx_util.c', ...
        'prolong_mex.c', ...
        'residual_mex.c', ...
        'restrict_mex.c', ...
    };

    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DEFINES, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fmg_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = { ...
        'fmg_mex.c', ...
        'boundary_mask_mex.c', ...
        'coarsen_grid_mex.c', ...
        'correct_mex.c', ...
        'gauss_seidel_mex.c', ...
        'mx_blas.c', ...
        'mx_util.c', ...
        'prolong_mex.c', ...
        'residual_mex.c', ...
        'restrict_mex.c', ...
    };

    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DEFINES, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % mg_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = { ...
        'mg_mex.c', ...
        'boundary_mask_mex.c', ...
        'coarsen_grid_mex.c', ...
        'correct_mex.c', ...
        'gauss_seidel_mex.c', ...
        'mx_blas.c', ...
        'mx_util.c', ...
        'prolong_mex.c', ...
        'residual_mex.c', ...
        'restrict_mex.c', ...
    };

    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DEFINES, LIBS);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % coarsen_grid_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'coarsen_grid_mex.c'};
    DMEX = '-DCOARSEN_GRID_MEX';
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % restrict_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'restrict_mex.c'};
    DMEX = '-DRESTRICT_MEX';
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % prolong_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'prolong_mex.c'};
    DMEX = '-DPROLONG_MEX';
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % correct_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'correct_mex.c'};
    DMEX = '-DCORRECT_MEX';
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gauss_seidel_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'gauss_seidel_mex.c'};
    DMEX = ['-DGAUSS_SEIDEL_MEX', ' ', DGAUSS_SEIDEL_RED_BLACK];
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % residual_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'residual_mex.c'};
    DMEX = '-DRESIDUAL_MEX';
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % lapmg_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'lapmg_mex.c'};
    DMEX = '-DLAPMG_MEX';
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % boundary_mask_l1_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'boundary_mask_l1_mex.c'};
    DMEX = '-DBOUNDARY_MASK_L1_MEX';
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % boundary_mask_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'boundary_mask_mex.c'};
    DMEX = '-DBOUNDARY_MASK_MEX';
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % norm2_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'norm2_mex.c'};
    DMEX = '-DNORM2_MEX';
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DMEX, LIBS);

end


function [] = make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, DEFINES, LIBS)

    try
        fprintf('Compiling %s\n', CFILES{1});
        eval([ ...
            'mex ', ...
            'CFLAGS=', '''', CFLAGS, ''' ', ...
            'COPTIMFLAGS=', '''', COPTIMFLAGS, ''' ', ...
            'LDFLAGS=', '''', LDFLAGS, ''' ', ...
            'LDOPTIMFLAGS=', '''', LDOPTIMFLAGS, ''' ', ...
            DEFINES, ' ', ...
            LIBS, ' ', ...
            strjoin(CFILES)
        ]);
        fprintf('\n');

    catch ex
        fprintf('\n')
        warning(getReport(ex));
        fprintf('\n\n')
    end

end
