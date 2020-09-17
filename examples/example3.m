% add QSM.m to path
run('/path/to/QSM.m/addpathqsm.m');

% NIFTI
filename = '/path/to/file.nii/.gz';
outpath = '/path/to/save/results';

% Constants for normalizing phase
B0 = 3;
GYRO = 267.513;


% Load dataset
nii = loadNii(filename);
mag = nii.img(:,:,:,:,1);
phas = nii.img(:,:,:,:,2);

% if magnitude and phase stored in separate files
%  nii = loadNii(filenameMag);
%  mag = nii.img;
%  nii = loadNii(filenamePhase);
%  phas = nii.img;

vsz = nii.hdr.dime.pixdim(2:4);
% echo times have to be supplied manually as it might not be stored in the
% nifti header. for single echo nifti files TE might be stored here:
TE = nii.hdr.dime.pixdim(5);
% bdir is not stored in nifti headers
bdir = [0, 0, 1];

% brain extraction using FSL's bet. won't work on windows.
mask0 = generateMask(mag(:,:,:,end), vsz, '-m -n -f 0.5');

% erode mask to deal with boundary inconsistencies during brain extraction
mask1 = erodeMask(mask0, 5);

% unwrap phase + background field removing
uphas = unwrapLaplacian(phas, mask1, vsz);

% convert units
for t = 1:size(uphas, 4)
    uphas(:,:,:,t) = uphas(:,:,:,t) ./ (B0 * GYRO * TEs(t));
end

% remove non-harmonic background fields
P = fitPoly3d(uphas, 4, mask1, vsz);
fl = uphas - mask1.*P;

% dipole inversion
x = rts(fl, mask1, vsz, bdir);

% save mat-file
save qsm.mat mag phas uphas x mask1 bdir vsz TEs

% or nifti
saveNii(fullfile(outpath, 'chi.nii'), x, vsz);

% view images
Plotter({mag})
Plotter({phas})
Plotter({uphas, fl}, 'contrast', [-0.05, 0.05], 'subsize', [2, size(fl, 4)])
Plotter({x}, 'contrast', [-0.15, 0.15])
