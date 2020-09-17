#pragma once

#include <inttypes.h>
#include "matrix.h"


extern void mx_gauss_seidel(mxArray *mxv,
                            const mxArray *mxf, const mxArray *mxG,
                            const mxArray *mxh,
                            int32_t iter, const uint8_t reverse);
