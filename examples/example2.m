% add QSM.m to path
run('/path/to/QSM.m/addpathqsm.m');

% DICOM
filename = '/path/to/dicom/directory';
outpath = '/path/to/save/results';

% Constants for normalizing phase
B0 = 3;
GYRO = 267.513;


% Load dataset
[hdr, data] = dicom(filename);
mag = data(:,:,:,:,1);
phas = data(:,:,:,:,2);

% if magnitude and phase stored in separate files
%  [hdr, data] = dicom(filenameMag);
%  mag = data(:,:,:,:,1);
%  [hdr, data] = dicom(filenamePhase);
%  phas = data(:,:,:,:,2);

vsz = hdr.vsz;
TEs = hdr.TEs;
bdir = hdr.bdir;

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
[fl, mask1] = vsharp(uphas, mask1, vsz, 9:-2*max(vsz):2*max(vsz), 0.05);

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
