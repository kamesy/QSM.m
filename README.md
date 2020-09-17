# QSM.m

QSM.m is a MATLAB toolbox for quantitative susceptibility mapping (QSM).

## Getting started

### Prerequisites

* FSL's bet for automatic brain extraction
* [optional] FSL's prelude for phase unwrapping

### Installation

Download or clone the repository, build mex files, add toolbox to MATLAB path.

```matlab
>> run('/path/to/QSM.m/make.m')         % build mex files
>> run('/path/to/QSM.m/addpathqsm.m')   % add toolbox to MATLAB path
```

## Usage

See [examples](examples).

```bash
QSM.m
├── examples
├── src
│   ├── bgremove            # background field removing methods
│   ├── inversion           # dipole inversion methods
│   │   ├── masks_weights
│   │   └── mex
│   ├── io                  # dicom, par/xml rec, nifti
│   │   └── dicomSorter
│   ├── unwrap              # phase unwrapping methods
│   │   └── mex
│   └── utils
│       ├── error_metrics
│       ├── fd              # finite difference operators
│       │   └── mex
│       ├── kernels         # dipole, smv
│       └── poisson_solver  # multigrid
│           ├── mex
│           └── test
└── third_party
    ├── lsmr
    ├── lsqrSOL
    └── NIfTI
```
