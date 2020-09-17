% add QSM.m to path
run('/path/to/QSM.m/addpathqsm.m');

% Philips PAR/REC or XML/REC
filename = '/path/to/input/file.par/or/file.xml';
outpath = '/path/to/save/results';

% Constants for normalizing phase
B0 = 3;
GYRO = 267.513;


% Load dataset
[hdr, data] = parxrec(filename);
mag = data(:,:,:,:,1);    % for multi-echo data. for single-echo:
phas = data(:,:,:,:,2);   % data(:,:,:,1/2), 4th dim is echoes

vsz = [hdr.images(1).Pixel_Spacing, ...
       hdr.images(1).Slice_Thickness + hdr.images(1).Slice_Gap];
TEs = 1e-3 * unique([hdr.images(:).Echo_Time]);

% computing bdir from philips header is experimental.
ang = [hdr.images(1).Angulation_AP, ...
       hdr.images(1).Angulation_FH, ...
       hdr.images(1).Angulation_RL];
[z, y, x] = sph2cart(pi/180 * ang(1), pi/180*ang(3), 1);
bdir = [x, y, z];

% or
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
fl = pdf(uphas, mask1, vsz, [], bdir, [], 1e-5, ceil(sqrt(numel(mask1))), 0);

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
