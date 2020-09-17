function [mask] = generateMask(mag, vsz, betargs)
%GENERATEMASK Automatic brain extraction using FSL's bet.
%
%   [mask] = GENERATEMASK(mag, vsz, [betargs]);
%
%   Inputs
%   ------
%       mag         magnitude image used as input for bet.
%       vsz         voxel size.
%
%       betargs     string containing command line arguments for bet.
%                   Default = '-m -n -f 0.5'.
%
%   Outputs
%   -------
%       mask        logical array containing binary mask.

    narginchk(2, 3);

    if nargin < 3, betargs = '-m -n -f 0.5'; end

    validateinputs(mag, vsz, betargs);

    mask = generateMask_(mag, vsz, betargs);

end


function [mask] = generateMask_(mag, vsz, betargs)

    magfile = [tempname, '.nii'];
    maskfile = strrep(magfile, '.nii', '_mask.nii.gz');

    saveNii(magfile, mag, vsz);

    ex = [];
    try
        % bet appends '_mask' to output filename regardless of user choice
        [~, cmdout] = unix(['bet ', magfile, ' ', magfile, ' ', betargs]);
        nii = loadNii(maskfile);
        mask = logical(nii.img);
    catch ex
        % void
    end

    if exist(magfile, 'file') == 2
        delete(magfile)
    end

    if exist(maskfile, 'file') == 2
        delete(maskfile)
    end

    if ~isempty(ex)
        fprintf(2, '%s\n', cmdout);
        rethrow(ex);
    end

end


function [] = validateinputs(mag, vsz, betargs)

    if ndims(mag) < 4
        nd = 3;
    else
        nd = 4;
    end

    classes = {'numeric'};
    attributes = {'real', 'ndims', nd};
    validateattributes(mag, classes, attributes, mfilename, 'mag', 1);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 2);

    classes = {'char'};
    validateattributes(betargs, classes, {}, mfilename, 'betargs', 3);

end
