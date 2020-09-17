#pragma once

#include "matrix.h"


extern void mx_lapmg(mxArray *mxdu,
                     const mxArray *mxu, const mxArray *mxG,
                     const mxArray *mxh);
