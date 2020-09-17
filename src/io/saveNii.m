function [nii] = saveNii(filename, img, vsz)
%SAVENII Save NIfTI dataset.
%   Wrapper for Jimmy Shen's NIfTI and ANALYZE toolbox.
%
%   [nii] = SAVENII(filename, img, [vsz]);
%
%   See also MAKE_NII, SAVE_NII, LOAD_NII, LOADNII

    narginchk(2, 3);

    if nargin < 3, vsz = [1, 1, 1]; end


    [pathstr, ~, ~] = fileparts(filename);

    if exist(pathstr, 'dir') ~= 7
        [status, msg] = mkdir(pathstr);
        if status ~= 1
            error(msg);
        end
    end

    nii = make_nii(img, vsz);
    save_nii(nii, filename);

end
