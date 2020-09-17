#pragma once

#include "matrix.h"


extern void mx_residual(mxArray *mxr,
                        const mxArray *mxf,
                        const mxArray *mxx,
                        const mxArray *mxG,
                        const mxArray *mxh);
