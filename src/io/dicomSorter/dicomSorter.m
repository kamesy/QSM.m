function [paths] = dicomSorter(dirname, outdir, mv)
%DICOMSORTER Sort dicom files into separate folders.
%
%   [paths] = dicomSorter(dirname, [outdir], [mv]);
%
%   Inputs
%   ------
%       dirname     input directory containing dicom files.
%       outdir      output directory, folders will be created here.
%                   Default: dirname.
%       mv          boolean flag.
%                   false (default): copy files, true: move files.
%
%   Outputs
%   -------
%       paths       cell array containing paths of all folders.
%
%   Notes
%   -----
%       MATLAB's DICOMINFO is extremely slow. A C implementation would
%       substantially speed things up.
%
%   See also DICOMINFO, DICOM

    narginchk(1, 3);

    if nargin < 3, mv = false; end
    if nargin < 2 || isempty(outdir), outdir = dirname; end

    if exist(dirname, 'dir') ~= 7
        error('directory does not exist: %s', dirname);
    end

    if exist(outdir, 'dir') ~= 7
        [status, msg] = mkdir(outdir);
        if status ~= 1
            error(msg);
        end
    end

    files = getFiles(dirname);
    scans = getScans(files);

    k = keys(scans);

    for ii = 1:length(k)
        paths{ii} = processScan(outdir, k{ii}, scans(k{ii}), mv); %#ok<AGROW>
    end

end % dicomSorter


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pathstr] = processScan(outdir, name, files, mv)

    pathstr = fullfile(outdir, strrep(name, ' ', '_'));

    if exist(pathstr, 'dir') ~= 7
        [status, msg] = mkdir(pathstr);
        if status ~= 1
            warning(msg);
            pathstr = [];
            return;
        end
    end

    if mv
        fprintf('Moving scan %s\n', name);
        for ii = 1:length(files)
            [~, fn, ext] = fileparts(files{ii});
            [status, msg] = movefile(files{ii}, fullfile(pathstr, [fn, ext]));
            if status ~= 1
                warning(msg);
            end
        end
    else
        fprintf('Copying scan %s\n', name);
        for ii = 1:length(files)
            [~, fn, ext] = fileparts(files{ii});
            [status, msg] = copyfile(files{ii}, fullfile(pathstr, [fn, ext]));
            if status ~= 1
                warning(msg);
            end
        end
    end

end % processScan


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [scans] = getScans(files)

    PATH = fileparts(mfilename('fullpath'));
    DCMDICT = [PATH, '/dicom.txt'];

    scans = containers.Map('KeyType', 'char', 'ValueType', 'any');
    hdrs{length(files), 1} = struct();

    parfor ii = 1:length(files)
        hdrs{ii} = dicominfo(files{ii}, 'dictionary', DCMDICT);
    end

    for ii = 1:length(files)
        hdr = hdrs{ii};

        key = '';
        if isfield(hdr, 'ProtocolName')
            key = [key, strrep(deblank(hdr.ProtocolName), ' ', '_'), ' '];
        end
        if isfield(hdr, 'SeriesDescription')
            key = [key, strrep(deblank(hdr.SeriesDescription), ' ', '_'), ' '];
        end
        if isfield(hdr, 'SeriesNumber')
            key = [key, num2str(hdr.SeriesNumber), ' '];
        end
        if isfield(hdr, 'AcquisitionNumber')
            key = [key, num2str(hdr.AcquisitionNumber)];
        end

        key = strrep(deblank(key), ' ', '_');

        if ~isKey(scans, key)
            scans(key) = files(ii);
        else
            tmp = scans(key);
            scans(key) = [tmp, files(ii)];
        end
    end

end % getScans


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [files] = getFiles(dirname)

    if ~isequal(dirname(end), filesep)
        dirname = [dirname, filesep];
    end

    files{10000,1} = '';
    q = java.util.LinkedList();
    add(q, dirname);

    idx = 0;
    while ~isEmpty(q)
        p = remove(q);
        D = dir(p);
        for ii = 1:length(D)
            if D(ii).isdir
                if strcmp(D(ii).name, '.') || strcmp(D(ii).name, '..')
                    continue;
                else
                    add(q, [p, D(ii).name, filesep]);
                end
            else
                file = fullfile(p, D(ii).name);
                if isdicom(file)
                    idx = idx + 1;
                    files{idx} = file;
                end
            end
        end
    end

    if idx ~= length(files)
        files(idx+1:end,:) = [];
    end

end % getFiles
