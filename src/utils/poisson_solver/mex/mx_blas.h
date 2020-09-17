#pragma once

#include "matrix.h"


/* Level 1 BLAS */
extern void mx_swap(mxArray *mxx, mxArray *mxy);
extern void mx_scal(const double a, mxArray *mxx);
extern void mx_copy(const mxArray *mxx, mxArray *mxy);
extern void mx_axpy(const double a, const mxArray *mxx, mxArray *mxy);

extern double mx_dot(const mxArray *mxx, const mxArray *mxy);
extern double mx_nrm2(const mxArray *mxx);
extern double mx_asum(const mxArray *mxx);
