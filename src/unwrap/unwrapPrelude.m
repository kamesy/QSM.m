function [uphas] = unwrapPrelude(phas, mag, mask, vsz, preludeArgs)
%UNWRAPPRELUDE Prelude phase unwrapping.
%
%   [uphas] = UNWRAPPRELUDE(phas, mag, mask, vsz, [preludeArgs]);
%
%   See also UNWRAPLAPLACIAN

    narginchk(4, 5);

    if nargin < 5, preludeArgs = ''; end

    validateinputs(phas, mag, mask, vsz, preludeArgs);

    uphas = zeros(size(phas), 'like', phas);
    for t = 1:size(phas, 4)
        uphas(:,:,:,t) = ...
            unwrap_(phas(:,:,:,t), mag(:,:,:,t), mask, vsz, preludeArgs);
    end

end


function [uphas] = unwrap_(phas, mag, mask, vsz, preludeArgs)

    T = class(phas);
    mag = cast(mag, T);
    mask = cast(mask, T);

    ofile = [tempname, '.nii.gz'];
    pfile = strrep(ofile, '.nii.gz', '_phas.nii');
    afile = strrep(ofile, '.nii.gz', '_mag.nii');
    mfile = strrep(ofile, '.nii.gz', '_mask.nii');

    saveNii(pfile, phas, vsz);
    saveNii(afile, mag, vsz);
    saveNii(mfile, mask, vsz);

    ex = [];
    try
        [~, cmdout] = unix([ ...
            'prelude', ...
            ' -p ', pfile, ...
            ' -a ', afile, ...
            ' -m ', mfile, ...
            ' -o ', ofile, ...
            '    ', preludeArgs ...
        ]);
        nii = loadNii(ofile);
        uphas = cast(nii.img, T);
    catch ex
        % void
    end

    if exist(pfile, 'file') == 2
        delete(pfile);
    end

    if exist(afile, 'file') == 2
        delete(afile);
    end

    if exist(mfile, 'file') == 2
        delete(mfile);
    end

    if exist(ofile, 'file') == 2
        delete(ofile);
    end

    if ~isempty(ex)
        fprintf(2, '%s\n', cmdout);
        rethrow(ex);
    end

end


function [] = validateinputs(phas, mag, mask, vsz, preludeArgs)

    sz = size(phas);

    if ndims(phas) < 4
        nd = 3;
    else
        nd = 4;
    end

    classes = {'numeric'};
    attributes = {'real', 'ndims', nd};
    validateattributes(phas, classes, attributes, mfilename, 'phas', 1);

    classes = {'numeric'};
    attributes = {'real', 'size', sz};
    validateattributes(mag, classes, attributes, mfilename, 'mag', 2);

    classes = {'logical', 'numeric'};
    attributes = {'real', 'ndims', 3, 'size', sz(1:3), 'binary'};
    validateattributes(mask, classes, attributes, mfilename, 'mask', 3);

    classes = {'numeric'};
    attributes = {'real', 'vector', 'numel', 3, 'finite', '>', 0};
    validateattributes(vsz, classes, attributes, mfilename, 'vsz', 4);

    classes = {'char'};
    validateattributes(preludeArgs, classes, {}, mfilename, 'preludeArgs', 5);

end
