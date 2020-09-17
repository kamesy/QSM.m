function [hdr, data] = dicom(dirname)
%DICOM Load dicom dataset.
%
%   [hdr, data] = DICOM(dirname);
%
%   Inputs
%   ------
%       dirname     directory containing dicom files.
%
%   Outputs
%   -------
%       hdr.dcms    cell array containing all dicom hdrs and filenames:
%                   hdr.dcms{1}.hdr, hdr.dcms{1}.file;
%       hdr.vsz     voxel size.
%       hdr.bdir    B field direction.
%       hdr.TEs     echo times in seconds.
%       data        5d array with dimensions [x, y, z, t, mag(1)/phas(2)].
%
%   Notes
%   -----
%       MATLAB's DICOMINFO is extremely slow. A C implementation would
%       substantially speed things up.
%
%   See also DICOMINFO, DICOMREAD, DICOMSORTER

    narginchk(1, 1);

    if exist(dirname, 'dir') ~= 7
        error('directory does not exist: %s', dirname);
    end

    files = getFiles(dirname);
    scans = getScans(files);

    if length(scans) > 1
        key = chooseScan(scans);
    else
        key = keys(scans);
        key = key{1};
    end

    scan = scans(key);
    if ~isempty(strfind(lower(scan{1}.hdr.Manufacturer), 'philips'))
        [hdr, data] = loadPhilips(scan);
    elseif ~isempty(strfind(lower(scan{1}.hdr.Manufacturer), 'ge'))
        [hdr, data] = loadGE(scan);
    elseif ~isempty(strfind(lower(scan{1}.hdr.Manufacturer), 'siemens'))
        [hdr, data] = loadSiemens(scan);
    else
        error('manufacturer not supported: %s', scan{1}.hdr.Manufacturer);
    end

    ks = repmat({key}, size(hdr));
    [hdr.name] = ks{:};

end % dicom


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hdr, data] = loadPhilips(dcms)

    h1 = dcms{1}.hdr;

    sz  = double([h1.Columns, h1.Rows, h1.NumberOfSlices, h1.EchoTrainLength]);
    vsz = double([reshape(h1.PixelSpacing, 1, []), 0]);

    TEs  = zeros(1, sz(4));
    mag  = zeros(sz, 'single');
    phas = zeros(sz, 'single');

    minpos = 0;
    maxpos = 0;

    for ii = 1:length(dcms)
        h = dcms{ii}.hdr;
        f = dcms{ii}.file;

        slice = h.SliceNumber;
        echo  = h.EchoNumber;

        if ~TEs(echo)
            TEs(echo) = 1e-3 * double(h.EchoTime);
        end

        rs = double(h.RescaleSlope);
        ri = double(h.RescaleIntercept);
        ss = double(h.ScaleSlope);

        if (isfield(h, 'MRImageType') && strcmpi(h.MRImageType, 'M')) ...
            || (isfield(h, 'ComplexImageComponent') ...
                && ~isempty(strfind(lower(h.ComplexImageComponent), 'mag'))) ...
            || (isfield(h, 'ImageType') ...
                && ~isempty(strfind(lower(h.ImageType), '\\m\\')))

            if slice == 1
                minpos = h.ImagePositionPatient;
            elseif slice == sz(3)
                maxpos = h.ImagePositionPatient;
            end

            if slice == 2
                vsz(3) = norm(minpos - h.ImagePositionPatient);
            end

            mag(:,:,slice,echo) = single(transpose(dicomread(f)))/ss + ri/(rs*ss);

        elseif (isfield(h, 'MRImageType') && strcmpi(h.MRImageType, 'P')) ...
            || (isfield(h, 'ComplexImageComponent') ...
                && ~isempty(strfind(lower(h.ComplexImageComponent), 'ph'))) ...
            || (isfield(h, 'ImageType') ...
                && ~isempty(strfind(lower(h.ImageType), '\\p\\')))

            phas(:,:,slice,echo) = single(transpose(dicomread(f)))/ss + ri/(rs*ss);

        else
            error('image type not supported: %s', f);
        end

    end

    o = h1.ImageOrientationPatient;
    bdir = [o(3), o(6), o(1)*o(5)-o(2)*o(4)];

    hdr = struct('dcms', dcms, 'vsz', vsz, 'bdir', bdir, 'TEs', TEs);
    data = cat(5, mag, phas);

end % loadPhilips


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hdr, data] = loadGE(dcms)

    sliceLoc = [inf, -inf];
    minpos = 0;
    maxpos = 0;

    for ii = 1:length(dcms)
        h = dcms{ii}.hdr;

        if h.SliceLocation < sliceLoc(1)
            sliceLoc(1) = h.SliceLocation;
            minpos = h.ImagePositionPatient;
        end

        if h.SliceLocation > sliceLoc(2)
            sliceLoc(2) = h.SliceLocation;
            maxpos = h.ImagePositionPatient;
        end
    end

    h1 = dcms{1}.hdr;

    sz  = double([h1.Columns, h1.Rows, h1.LocationsInAcquisition, h1.NumberOfEchoes]);
    vsz = double([reshape(h1.PixelSpacing, 1, []), 0]);

    TEs = zeros(1, sz(4));

    re  = zeros(sz, 'single');
    im  = zeros(sz, 'single');
    mag = zeros(sz, 'single');
    phas = zeros(sz, 'single');

    for ii = 1:length(dcms)
        h = dcms{ii}.hdr;
        f = dcms{ii}.file;

        slice = round(norm(h.ImagePositionPatient-minpos)/h.SpacingBetweenSlices) + 1;
        echo  = h.EchoNumber;

        if ~TEs(echo)
            TEs(echo) = 1e-3 * double(h.EchoTime);
        end

        if isfield(h, 'RescaleSlope')
            rs = h.RescaleSlope;
        else
            rs = 1;
        end

        if isfield(h, 'RescaleIntercept')
            ri = h.RescaleIntercept;
        else
            ri = 0;
        end

        rs = single(rs);
        ri = single(ri);

        if h.ImageTypeGE == 2
            if slice == 2
                vsz(3) = norm(minpos - h.ImagePositionPatient);
            end
            re(:,:,slice,echo) = rs * transpose(single(dicomread(f))) + ri;
        elseif h.ImageTypeGE == 3
            im(:,:,slice,echo) = rs * transpose(single(dicomread(f))) + ri;
        elseif h.ImageTypeGE == 0
            mag(:,:,slice,echo) = rs * transpose(single(dicomread(f))) + ri;
        elseif h.ImageTypeGE == 1
            phas(:,:,slice,echo) = rs * transpose(single(dicomread(f))) + ri;
        else
            error('image type not supported: %s', f);
        end
    end

    re(:,:,1:2:end,:) = -re(:,:,1:2:end,:);
    im(:,:,1:2:end,:) = -im(:,:,1:2:end,:);

    if ~any(vec(mag))
        mag = sqrt(re.*re + im.*im);
    end

    if ~any(vec(phas))
        phas = atan2(im, re);
    end

    o = h1.ImageOrientationPatient;
    bdir = [o(3), o(6), o(1)*o(5)-o(2)*o(4)];

    hdr = struct('dcms', dcms, 'vsz', vsz, 'bdir', bdir, 'TEs', TEs);
    data = cat(5, mag, phas);

end % loadGE


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hdr, data] = loadSiemens(dcms)

    sliceLoc = [inf, -inf];
    minpos = 0;
    maxpos = 0;

    tmp = zeros(1, length(dcms), 'single');

    for ii = 1:length(dcms)
        h = dcms{ii}.hdr;

        tmp(ii) = single(h.SliceLocation);

        if h.SliceLocation < sliceLoc(1)
            sliceLoc(1) = h.SliceLocation;
            minpos = h.ImagePositionPatient;
        end

        if h.SliceLocation > sliceLoc(2)
            sliceLoc(2) = h.SliceLocation;
            maxpos = h.ImagePositionPatient;
        end
    end

    h1 = dcms{1}.hdr;

    sz  = double([h1.Columns, h1.Rows, length(unique(tmp)), h1.EchoTrainLength]);
    vsz = double([reshape(h1.PixelSpacing, 1, []), 0]);

    TEs  = zeros(1, sz(4));
    mag  = zeros(sz, 'single');
    phas = zeros(sz, 'single');

    if strcmpi(h1.MRAcquisitionType, '2D')
        iden = 1 / h1.SpacingBetweenSlices;
    else
        iden = 1 / h1.SliceThickness;
    end

    for ii = 1:length(dcms)
        h = dcms{ii}.hdr;
        f = dcms{ii}.file;

        slice = round(iden .* norm(h.ImagePositionPatient-minpos)) + 1;
        echo  = h.EchoNumber;

        if ~TEs(echo)
            TEs(echo) = 1e-3 * double(h.EchoTime);
        end

        if isfield(h, 'RescaleSlope')
            rs = double(h.RescaleSlope);
        else
            rs = 1;
        end

        if isfield(h, 'RescaleIntercept')
            ri = double(h.RescaleIntercept);
        else
            ri = 0;
        end

        if (isfield(h, 'MRImageType') && strcmpi(h.MRImageType, 'M')) ...
            || (isfield(h, 'ComplexImageComponent') ...
                && ~isempty(strfind(lower(h.ComplexImageComponent), 'mag'))) ...
            || (isfield(h, 'ImageType') ...
                && ~isempty(strfind(lower(h.ImageType), '\m\')))

            if slice == 2
                vsz(3) = norm(minpos - h.ImagePositionPatient);
            end

            mag(:,:,slice,echo) = rs * single(transpose(dicomread(f))) + ri;

        elseif (isfield(h, 'MRImageType') && strcmpi(h.MRImageType, 'P')) ...
            || (isfield(h, 'ComplexImageComponent') ...
                && ~isempty(strfind(lower(h.ComplexImageComponent), 'ph'))) ...
            || (isfield(h, 'ImageType') ...
                && ~isempty(strfind(lower(h.ImageType), '\p\')))

            phas(:,:,slice,echo) = rs * single(transpose(dicomread(f))) + ri;

        else
            error('image type not supported: %s', f);
        end

    end

    ex = [min(vec(phas)), max(vec(phas))];
    if ex(1) ~= -pi && ex(2) ~= pi
        if ex(2) ~= 0
            phas = pi .* phas ./ ex(2);
        end
    end

    o = h1.ImageOrientationPatient;
    bdir = [o(3), o(6), o(1)*o(5)-o(2)*o(4)];

    hdr = struct('dcms', dcms, 'vsz', vsz, 'bdir', bdir, 'TEs', TEs);
    data = cat(5, mag, phas);

end % loadSiemens


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
        if isempty(hdr) || isempty(hdr.Width) || isempty(hdr.Height)
            continue;
        end

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
            scans(key) = {struct('file', files{ii}, 'hdr', hdr)};
        else
            tmp = scans(key);
            scans(key) = [tmp, struct('file', files{ii}, 'hdr', hdr)];
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [key] = chooseScan(scans)

    K = keys(scans);
    n = numel(K);

    fprintf('\nThe following datasets were found in the directory\n\n');
    for ii = 1:n
        fprintf('\t%2d. %s\n', ii, K{ii});
    end
    fprintf('\n');

    idx = 0;

    while ~isnumeric(idx) || idx < 1 || idx > n
        idx = input('Enter number of dataset to process: ');
    end
    fprintf('\n');

    key = K{idx};

end % chooseScan
