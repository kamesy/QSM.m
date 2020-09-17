function [hdr, data] = parxrec(filename)
%PARXREC Load Philips PAR/XML REC dataset.
%
%   [hdr, data] = PARXREC(filename);
%
%   PARXREC parses a par/xml header into a MATLAB struct (hdr) and reads the
%   corresponding rec file (data).
%
%       hdr contains the full par/xml header without further processing.
%
%       data is the sorted, n-dimensional dataset stored in the rec file. The
%       dimensions of data are stacked according to the order of occurrence
%       in the par/xml header.
%
%   Example
%   -------
%       % load multi-echo GRE dataset: [x, y, z, echo, imagetype]
%       [hdr, data] = parxrec('dataset.PAR');
%       mag = data(:,:,:,:,1);
%       phas = data(:,:,:,:,2);
%
%       % extract voxelsize from hdr
%       vsz = [hdr.images(1).Pixel_Spacing, ...
%              hdr.images(1).Slice_Thickness + hdr.images(1).Slice_Gap];
%
%       % extract echotimes from hdr
%       TEs = 1e-3 * unique([hdr.images(:).Echo_Time]);
%
%       % extract B-field direction
%       ang = [hdr.images(1).Angulation_AP, ...
%              hdr.images(1).Angulation_FH, ...
%              hdr.images(1).Angulation_RL];
%       [z, y, x] = sph2cart(pi/180 * ang(1), pi/180*ang(3), 1);
%       bdir = [x, y, z];
%
%   Notes
%   -----
%       Parsing an XML header takes a long time due to MATLAB's slow XML
%       utility functions.

    narginchk(1, 1);
    loadData = nargout > 1;

    [hdrpath, recpath] = parseinputs(filename, loadData);

    [~, ~, ext] = fileparts(hdrpath);
    if strcmpi(ext, '.par')
        hdr = readpar(hdrpath);
    else
        hdr = readxml(hdrpath);
    end

    if loadData
        data = readrec(recpath, hdr);
    end

end % parxrec


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [data] = readrec(recpath, hdr)

    validateHdr(hdr)
    hdr = hdr.images;

    T = containers.Map({8, 16, 32, 64}, {'uint8', 'uint16', 'single', 'double'});

    nx       = double(hdr(1).Resolution_X);
    ny       = double(hdr(1).Resolution_Y);
    nz       = double(max([hdr.Slice]));
    echoes   = unique([hdr.Echo]);
    dynamics = unique([hdr.Dynamic]);
    phases   = unique([hdr.Phase]);
    bvals    = unique([hdr.BValue]);
    gorients = unique([hdr.Grad_Orient]);

    % Enumeration in .xml hdr, number in .par hdr
    if isa(hdr(1).Type, 'numeric')
        types  = containers.Map('KeyType', 'double', 'ValueType', 'double');
        types_ = unique([hdr.Type]);
        for ii = 1:length(types_)
            types(types_(ii)) = ii;
        end
    else
        types  = containers.Map('KeyType', 'char', 'ValueType', 'double');
        types_ = unique({hdr.Type});
        for ii = 1:length(types_)
            types(types_{ii}) = ii;
        end
    end

    if isa(hdr(1).Label_Type, 'numeric')
        labels  = containers.Map('KeyType', 'double', 'ValueType', 'double');
        labels_ = unique([hdr.Label_Type]);
        for ii = 1:length(labels_)
            labels(labels_(ii)) = ii;
        end
    else
        labels  = containers.Map('KeyType', 'char', 'ValueType', 'double');
        labels_ = unique({hdr.Label_Type});
        for ii = 1:length(labels_)
            labels(labels_{ii}) = ii;
        end
    end

    if isa(hdr(1).Sequence, 'numeric')
        seqs  = containers.Map('KeyType', 'double', 'ValueType', 'double');
        seqs_ = unique([hdr.Sequence]);
        for ii = 1:length(seqs_)
            seqs(seqs_(ii)) = ii;
        end
    else
        seqs  = containers.Map('KeyType', 'char', 'ValueType', 'double');
        seqs_ = unique({hdr.Sequence});
        for ii = 1:length(seqs_)
            seqs(seqs_{ii}) = ii;
        end
    end

    if length(unique([hdr.Pixel_Size])) == 1
        fd = dir(recpath);
        sz = (nx * ny * hdr(1).Pixel_Size/8 * length(hdr)) - fd.bytes;
        if sz < 0
            warning('REC file larger than inferred from hdr');
        elseif sz > 0
            warning('REC file smaller than inferred from hdr');
        end
    end

    data = zeros(...
        nx, ny, nz, ...
        length(echoes), ...
        length(dynamics), ...
        length(phases), ...
        length(bvals), ...
        length(gorients), ...
        length(labels), ...
        length(types), ...
        length(seqs), ...
        'single' ...
    );

    fid = fopen(recpath, 'rb', 'l');
    if (fid == -1)
        error('cannot open file: %s', recpath);
    end

    for ii = 1:length(hdr)
        img = hdr(ii);

        fseek(fid, img.Index * (nx*ny*img.Pixel_Size/8), 'bof');
        d = fread(fid, [nx, ny], T(img.Pixel_Size));

        if size(d,1) == nx && size(d,2) == ny
            i3  = img.Slice;
            i4  = img.Echo;
            i5  = img.Dynamic;
            i6  = img.Phase;
            i7  = img.BValue;
            i8  = img.Grad_Orient;
            i9  = labels(img.Label_Type);
            i10 = types(img.Type);
            i11 = seqs(img.Sequence);
            ss  = 1.0 / img.Scale_Slope;
            ri  = ss * img.Rescale_Intercept / img.Rescale_Slope;
            data(:,:,i3,i4,i5,i6,i7,i8,i9,i10,i11) = ss*d + ri;
        else
            fclose(fid);
            error('reading data, i = %d, idx = %d', ii, img.Index);
        end
    end
    fclose(fid);

    data = squeeze(data);

end % readrec


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hdr] = readpar(hdrpath)

    buf = fileread(hdrpath);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Series Info / General Information
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % lines starting with `.`
    str = strtrim(regexp(buf, '(?<=\n\.).*?\n', 'match'));
    str = strjoin(str, '\n');

    series.Patient_Name            = parSeriesInfo(str, 'Patient name', 0);
    series.Examination_Name        = parSeriesInfo(str, 'Examination name', 0);
    series.Protocol_Name           = parSeriesInfo(str, 'Protocol name', 0);

    Examination_Data_Time          = parSeriesInfo(str, 'Examination date/time', 0);
    Examination_Data_Time          = strtrim(strsplit(Examination_Data_Time, '/'));
    series.Examination_Date        = Examination_Data_Time{1};
    series.Examination_Time        = Examination_Data_Time{2};

    series.Series_Data_Type        = parSeriesInfo(str, 'Series Type', 0);
    series.Aquisition_Number       = parSeriesInfo(str, 'Acquisition nr', 1);
    series.Reconstruction_Number   = parSeriesInfo(str, 'Reconstruction nr', 1);
    series.Scan_Duration           = parSeriesInfo(str, 'Scan Duration', 1);
    series.Max_No_Phases           = parSeriesInfo(str, 'Max. number of cardiac phases', 1);
    series.Max_No_Echoes           = parSeriesInfo(str, 'Max. number of echoes', 1);
    series.Max_No_Slices           = parSeriesInfo(str, 'Max. number of slices', 1);
    series.Max_No_Dynamics         = parSeriesInfo(str, 'Max. number of dynamics', 1);
    series.Max_No_Mixes            = parSeriesInfo(str, 'Max. number of mixes', 1);
    series.Max_No_B_Values         = parSeriesInfo(str, 'Max. number of diffusion', 1);
    series.Max_No_Gradient_Orients = parSeriesInfo(str, 'Max. number of gradient', 1);
    series.No_Label_Types          = parSeriesInfo(str, 'Number of label types', 1);
    series.Patient_Position        = parSeriesInfo(str, 'Patient position', 0);
    series.Preparation_Direction   = parSeriesInfo(str, 'Preparation direction', 0);
    series.Technique               = parSeriesInfo(str, 'Technique', 0);

    Scan_Resolution_X_Y            = parSeriesInfo(str, 'Scan resolution', 1);
    series.Scan_Resolution_X       = Scan_Resolution_X_Y(1);
    series.Scan_Resolution_Y       = Scan_Resolution_X_Y(2);

    series.Scan_Mode               = parSeriesInfo(str, 'Scan mode', 0);
    series.Repetition_Times        = parSeriesInfo(str, 'Repetition time', 1);

    FOV_AP_FH_RL                   = parSeriesInfo(str, 'FOV', 1);
    series.FOV_AP                  = FOV_AP_FH_RL(1);
    series.FOV_FH                  = FOV_AP_FH_RL(2);
    series.FOV_RL                  = FOV_AP_FH_RL(3);

    series.Water_Fat_Shift         = parSeriesInfo(str, 'Water Fat', 1);

    Angulation_AP_FH_RL            = parSeriesInfo(str, 'Angulation', 1);
    series.Angulation_AP           = Angulation_AP_FH_RL(1);
    series.Angulation_FH           = Angulation_AP_FH_RL(2);
    series.Angulation_RL           = Angulation_AP_FH_RL(3);

    Off_Center_AP_FH_RL            = parSeriesInfo(str, 'Off Centre', 1);
    series.Off_Center_AP           = Off_Center_AP_FH_RL(1);
    series.Off_Center_FH           = Off_Center_AP_FH_RL(2);
    series.Off_Center_RL           = Off_Center_AP_FH_RL(3);

    series.Flow_Compensation       = parSeriesInfo(str, 'Flow', 1);
    series.Presaturation           = parSeriesInfo(str, 'Presaturation', 1);
    series.Phase_Encoding_Velocity = parSeriesInfo(str, 'Phase encoding', 1);
    series.MTC                     = parSeriesInfo(str, 'MTC', 1);
    series.SPIR                    = parSeriesInfo(str, 'SPIR', 1);
    series.EPI_factor              = parSeriesInfo(str, 'EPI factor', 1);
    series.Dynamic_Scan            = parSeriesInfo(str, 'Dynamic scan', 1);
    series.Diffusion               = parSeriesInfo(str, 'Diffusion    ', 1);
    series.Diffusion_Echo_Time     = parSeriesInfo(str, 'Diffusion echo time', 1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Image Info
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % lines starting with ` ` or digit
    str = regexp(buf, '(?<=\n)( |\d+).*?\n', 'match');
    str = cellfun(@(s) str2num(strtrim(s)), str, 'UniformOutput', false); %#ok<ST2NM>

    for ii = length(str):-1:1
        images(ii).Slice                     = str{ii}(1);
        images(ii).Echo                      = str{ii}(2);
        images(ii).Dynamic                   = str{ii}(3);
        images(ii).Phase                     = str{ii}(4);
        images(ii).BValue                    = str{ii}(42);
        images(ii).Grad_Orient               = str{ii}(43);
        images(ii).Label_Type                = str{ii}(49);
        images(ii).Type                      = str{ii}(5);
        images(ii).Sequence                  = str{ii}(6);
        images(ii).Index                     = str{ii}(7);
        images(ii).Pixel_Size                = str{ii}(8);
        images(ii).Scan_Percentage           = str{ii}(9);
        images(ii).Resolution_X              = str{ii}(10);
        images(ii).Resolution_Y              = str{ii}(11);
        images(ii).Rescale_Intercept         = str{ii}(12);
        images(ii).Rescale_Slope             = str{ii}(13);
        images(ii).Scale_Slope               = str{ii}(14);
        images(ii).Window_Center             = str{ii}(15);
        images(ii).Window_Width              = str{ii}(16);
        images(ii).Slice_Thickness           = str{ii}(23);
        images(ii).Slice_Gap                 = str{ii}(24);
        images(ii).Display_Orientation       = str{ii}(25);
        images(ii).fMRI_Status_Indication    = str{ii}(27);
        images(ii).Image_Type_Ed_Es          = str{ii}(28);
        images(ii).Pixel_Spacing             = str{ii}(29:30);
        images(ii).Echo_Time                 = str{ii}(31);
        images(ii).Dyn_Scan_Begin_Time       = str{ii}(32);
        images(ii).Trigger_Time              = str{ii}(33);
        images(ii).Diffusion_B_Factor        = str{ii}(34);
        images(ii).No_Averages               = str{ii}(35);
        images(ii).Image_Flip_Angle          = str{ii}(36);
        images(ii).Cardiac_Frequency         = str{ii}(37);
        images(ii).Min_RR_Interval           = str{ii}(38);
        images(ii).Max_RR_Interval           = str{ii}(39);
        images(ii).TURBO_Factor              = str{ii}(40);
        images(ii).Inversion_Delay           = str{ii}(41);
        images(ii).Contrast_Type             = str{ii}(44);
        images(ii).Diffusion_Anisotropy_Type = str{ii}(45);
        images(ii).Diffusion_AP              = str{ii}(46);
        images(ii).Diffusion_FH              = str{ii}(47);
        images(ii).Diffusion_RL              = str{ii}(48);
        images(ii).Angulation_AP             = str{ii}(17);
        images(ii).Angulation_FH             = str{ii}(18);
        images(ii).Angulation_RL             = str{ii}(19);
        images(ii).Offcenter_AP              = str{ii}(20);
        images(ii).Offcenter_FH              = str{ii}(21);
        images(ii).Offcenter_RL              = str{ii}(22);
        images(ii).Slice_Orientation         = str{ii}(26);
    end % for

    hdr.series = series;
    hdr.images = images;

end % readpar


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [val] = parSeriesInfo(str, id, tonum)

    str = regexp(str, sprintf('%s.*?(\n|$)', id), 'match', 'once');
    str = strtrim(strsplit(str, ': '));
    if tonum
        val = str2num(str{2}); %#ok<ST2NM>
    else
        val = str{2};
    end

end % parSeriesInfo


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hdr] = readxml(hdrpath)

    import javax.xml.xpath.*

    % XML tag names
    XML_HEADER           = 'PRIDE_V5';
    XML_SERIES_HEADER    = 'Series_Info';
    XML_IMAGE_ARR_HEADER = 'Image_Array';
    XML_IMAGE_HEADER     = 'Image_Info';
    XML_IMAGE_KEY_HEADER = 'Key';
    XML_ATTRIB_HEADER    = 'Attribute';

    try
        xDoc = xmlread(hdrpath);
    catch e
        error(e.message);
    end

    factory = XPathFactory.newInstance;
    xpath   = factory.newXPath;

    xp = xpath.compile('distinct-values(//*/name())');
    nodes = xp.evaluate(xDoc, XPathConstants.NODESET);

    if ~(nodes.contains(XML_HEADER) ...
            && nodes.contains(XML_SERIES_HEADER) ...
            && nodes.contains(XML_IMAGE_ARR_HEADER) ...
            && nodes.contains(XML_IMAGE_HEADER) ...
            && nodes.contains(XML_IMAGE_KEY_HEADER) ...
            && nodes.contains(XML_ATTRIB_HEADER))
        error('XMLREC tags not found in %s', hdrpath);
    end

    nodes = xDoc.getElementsByTagName(XML_SERIES_HEADER) ...
                .item(0).getElementsByTagName(XML_ATTRIB_HEADER);
    hdr.series = nodes2struct(nodes);

    nodes = xDoc.getElementsByTagName(XML_IMAGE_HEADER);

    for ii = 0:nodes.getLength()-1
        tmp = nodes.item(ii).getElementsByTagName(XML_ATTRIB_HEADER);
        hdr.images(ii+1) = nodes2struct(tmp);
    end

end % readxml


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [s] = nodes2struct(nodes)

    N = nodes.getLength() - 1;
    [c, f] = deal(cell(1, N+1));

    for ii = 0:N
        tmp = nodes.item(ii);
        val = cell(tmp.getTextContent());

        f(ii+1) = cell(tmp.getAttribute('Name'));
        type    = cell(tmp.getAttribute('Type'));
        c{ii+1} = convertCellToType(val, type);
    end

    f = strrep(f, ' ', '_');
    s = cell2struct(c, f, 2);

end % nodes2struct


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = convertCellToType(val, type)

    val = strtrim(val);
    val = val{:};

    switch type{:}
        case {'Double', 'Int32', 'Float', 'UInt16', 'Int16', 'UInt32'}
            val = sscanf(val, '%f')';
    end

    %{
    switch type{:}
        case 'Double'
            val = sscanf(val{:}, '%f');
        case 'Int32'
            val = int32(sscanf(val{:}, '%f'));
        case {'String', 'Date', 'Time', 'Boolean', 'Enumeration'}
            val = sscanf(val{:}, '%s');
        case 'Float'
            val = single(sscanf(val{:}, '%f'));
        case 'UInt16'
            val = uint16(sscanf(val{:}, '%f'));
        case 'Int16'
            val = int16(sscanf(val{:}, '%f'));
        case 'UInt32'
            val = uint32(sscanf(val{:}, '%f'));
        otherwise
            val = sscanf(val{:}, '%s');
    end
    %}

end % convertCellToType


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = validateHdr(hdr)

    series = hdr.series;
    images = hdr.images;

    %
    % Check whether data size in series info is consistent with image info
    %

    str = 'maximum number of ';
    if series.Max_No_Slices ~= max([images.Slice])
        printWarning1([str, 'slices'], series.Max_No_Slices, max([images.Slice]));
    end

    if series.Max_No_Echoes ~= max([images.Echo])
        printWarning1([str, 'echoes'], series.Max_No_Echoes, max([images.Echo]));
    end

    if series.Max_No_Dynamics ~= max([images.Dynamic])
        printWarning1([str, 'dynamics'], series.Max_No_Dynamics, max([images.Dynamic]));
    end

    if series.Max_No_Phases ~= max([images.Phase])
        printWarning1([str, 'phases'], series.Max_No_Phases, max([images.Phase]));
    end

    if series.Max_No_B_Values ~= max([images.BValue])
        printWarning1([str, 'bvalues'], series.Max_No_B_Values, max([images.BValue]));
    end

    if series.Max_No_Gradient_Orients ~= max([images.Grad_Orient])
        printWarning1([str, 'gradient orientations'], ...
            series.Max_No_Gradient_Orients, max([images.Grad_Orient]));
    end

    if isa(images(1).Label_Type, 'numeric') && ...
            (series.No_Label_Types+1 ~= max([images.Label_Type]))
        printWarning1([str, 'label types'], series.No_Label_Types, max([images.Label_Type]));
    end

    if isa(images(1).Label_Type, 'char') && ...
            (series.No_Label_Types+1 ~= length(unique({images.Label_Type})))
        printWarning1([str, 'label types'], ...
            series.No_Label_Types, length(unique({images.Label_Type})));
    end

    function printWarning1(field, val1, val2)
        warning('%s not matching: %d (series info) vs. %d (image info)', ...
            field, val1, val2);
    end


    %
    % Non-exhaustive image info check
    %

    str = 'inconsistent number of ';
    if max([images.Slice]) ~= length(unique([images.Slice]))
        printWarning2([str, 'slices'], ...
            max([images.Slice]), length(unique([images.Slice])));
    end

    if max([images.Echo]) ~= length(unique([images.Echo]))
        printWarning2([str, 'echoes'], ...
            max([images.Echo]), length(unique([images.Echo])));
    end

    if max([images.Dynamic]) ~= length(unique([images.Dynamic]))
        printWarning2([str, 'dynamics'], ...
            max([images.Dynamic]), length(unique([images.Dynamic])));
    end

    if max([images.Phase]) ~= length(unique([images.Phase]))
        printWarning2([str, 'phases'], ...
            max([images.Phase]), length(unique([images.Phase])));
    end

    if max([images.BValue]) ~= length(unique([images.BValue]))
        printWarning2([str, 'bvalues'], ...
            max([images.BValue]), length(unique([images.BValue])));
    end

    if max([images.Grad_Orient]) ~= length(unique([images.Grad_Orient]))
        printWarning2([str, 'gradient orientations'], ...
            max([images.Grad_Orient]), length(unique([images.Grad_Orient])));
    end

    if isa(images(1).Label_Type, 'numeric') && ...
            (max([images.Label_Type]) ~= length(unique([images.Label_Type])))
        printWarning2([str, 'label types'], ...
            max([images.Label_Type]), length(unique([images.Label_Type])));
    end

    if max([images.Index]) + 1 ~= length(images)
        printWarning2([str, 'indices'], max([images.Index]) + 1, length(images));
    end

    function printWarning2(field, val1, val2)
        warning('%s: %d (max) vs. %d (length(unique))', field, val1, val2);
    end

    if length(unique([images.Pixel_Size])) ~= 1
        warning('multiple values for Pixel Size')
    end

    if length(unique([images.Resolution_X])) ~= 1
        error('multiple values for Resolution X')
    end

    if length(unique([images.Resolution_Y])) ~= 1
        error('multiple values for Resolution Y')
    end

    if length(unique(cellfun(@(x) x(1), {images.Pixel_Spacing}))) ~= 1
        warning('multiple voxel sizes in x')
    end

    if length(unique(cellfun(@(x) x(2), {images.Pixel_Spacing}))) ~= 1
        warning('multiple voxel sizes in y')
    end

    if length(unique([images.Slice_Thickness])) ~= 1
        warning('multiple values for Slice Thickness')
    end

    if length(unique([images.Slice_Gap])) ~= 1
        warning('multiple values for Slice Gap')
    end

end % validateHdr


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hdrpath, recpath] = parseinputs(filename, loadData)

    [pn, fn, ext] = fileparts(filename);

    if isempty(ext)
        [filename, ~] = parseinputs([filename, '.REC'], loadData);
        [hdrpath, recpath] = parseinputs(filename, loadData);
        return;
    end

    switch lower(ext)
        case {'.par', '.xml'}
            hdrpath = filename;
            if loadData
                recpath = fullfile(pn, fn);
                if (exist([recpath, '.REC'], 'file') == 2)
                    recpath = [recpath, '.REC'];
                elseif (exist([recpath, '.rec'], 'file') == 2)
                    recpath = [recpath, '.rec'];
                else
                    error('.REC file not found: %s', filename);
                end
            else
                recpath = [];
            end

        case {'.rec'}
            recpath = filename;
            hdrpath = fullfile(pn, fn);
            if (exist([hdrpath, '.PAR'], 'file') == 2)
                hdrpath = [hdrpath, '.PAR'];
            elseif (exist([hdrpath, '.XML'], 'file') == 2)
                hdrpath = [hdrpath, '.XML'];
            elseif (exist([hdrpath, '.par'], 'file') == 2)
                hdrpath = [hdrpath, '.par'];
            elseif (exist([hdrpath, '.xml'], 'file') == 2)
                hdrpath = [hdrpath, '.xml'];
            else
                error('.PAR/.XML file not found: %s', filename);
            end

        otherwise
            error('file extension not supported: %s', filename);
    end

end % parseinputs
