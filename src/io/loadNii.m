function [nii] = loadNii(filename)
%LOADNII Load NIfTI dataset.
%   Wrapper for Jimmy Shen's NIfTI and ANALYZE toolbox.
%
%   [nii] = LOADNII(filename);
%
%   See also LOAD_NII, MAKE_NII, SAVE_NII, SAVENII

    narginchk(1, 1);

    if exist(filename, 'file') == 2
        [~, ~, ext] = fileparts(filename);
        if strcmpi(ext, '.gz')
            gunzip(filename);
            nii = load_nii(filename(1:end-3));
            delete(filename(1:end-3));
        elseif strcmpi(ext, '.nii')
            nii = load_nii(filename);
        else
            error('wrong file extension %s', filename);
        end
    else
        error('file %s does not exist', filename);
    end

end
