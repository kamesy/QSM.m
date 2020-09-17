#pragma once

#include "matrix.h"


extern void     mx_zero(mxArray *mxx);
extern mxArray *mx_pad_boundary(const mxArray *mxx);
extern mxArray *mx_unpad_boundary(const mxArray *mxx);
