function [] = make(openmp)

    if nargin < 1
        openmp = 1;
    end

    CFLAGS = '$CFLAGS -Wall -Wextra -pedantic -std=c11 -fomit-frame-pointer';
    COPTIMFLAGS = '$COPTIMFLAGS -O2';

    LDFLAGS = '$LDFLAGS';
    LDOPTIMFLAGS = '$LDOPTIMFLAGS -O2';

    LIBS = '';

    if openmp
        CFLAGS = [CFLAGS, ' ', '-fopenmp'];
        LDFLAGS = [LDFLAGS, ' ', '-fopenmp'];
        if isunix && ~ismac
            LIBS = [LIBS, ' ', '-lgomp'];
        end
    else
        CFLAGS = [CFLAGS, ' ', '-Wno-unknown-pragmas'];
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gradf_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'gradf_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    CFILES = {'gradfm_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gradb_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'gradb_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    CFILES = {'gradbm_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gradc_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'gradc_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    CFILES = {'gradcm_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gradf_adj_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'gradf_adj_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    CFILES = {'gradfm_adj_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gradb_adj_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'gradb_adj_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    CFILES = {'gradbm_adj_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % lap_mex.c
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CFILES = {'lap1_mex.c'};
    make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS);

end


function [] = make_(CFILES, CFLAGS, COPTIMFLAGS, LDFLAGS, LDOPTIMFLAGS, LIBS)

    try
        fprintf('Compiling %s\n', CFILES{1});
        eval([ ...
            'mex ', ...
            'CFLAGS=', '''', CFLAGS, ''' ', ...
            'COPTIMFLAGS=', '''', COPTIMFLAGS, ''' ', ...
            'LDFLAGS=', '''', LDFLAGS, ''' ', ...
            'LDOPTIMFLAGS=', '''', LDOPTIMFLAGS, ''' ', ...
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
