#include <blas.h>
#include <inttypes.h>
#include "mex.h"
#include "mx_blas.h"


/* Level 1 BLAS */
void
mx_swap(mxArray *mxx, mxArray *mxy)
{
    const ptrdiff_t incx = 1;
    const ptrdiff_t incy = 1;
    const ptrdiff_t n = (const ptrdiff_t)mxGetNumberOfElements(mxx);

    if (mxIsSingle(mxx)) {
        float *x = (float *)mxGetData(mxx);
        float *y = (float *)mxGetData(mxy);
        sswap(&n, x, &incx, y, &incy);

    } else if (mxIsDouble(mxx)) {
        double *x = (double *)mxGetData(mxx);
        double *y = (double *)mxGetData(mxy);
        dswap(&n, x, &incx, y, &incy);

    } else {
        mexErrMsgTxt("not implemented");
    }

    return;
} /* mx_swap() */


void
mx_scal(const double a, mxArray *mxx)
{
    const ptrdiff_t incx = 1;
    const ptrdiff_t n = (const ptrdiff_t)mxGetNumberOfElements(mxx);

    if (mxIsSingle(mxx)) {
        const float af = (const float)a;
        float *x = (float *)mxGetData(mxx);
        sscal(&n, &af, x, &incx);

    } else if (mxIsDouble(mxx)) {
        double *x = (double *)mxGetData(mxx);
        dscal(&n, &a, x, &incx);

    } else {
        mexErrMsgTxt("not implemented");
    }

    return;
} /* mx_scal() */


void
mx_copy(const mxArray *mxx, mxArray *mxy)
{
    const ptrdiff_t incx = 1;
    const ptrdiff_t incy = 1;
    const ptrdiff_t n = (const ptrdiff_t)mxGetNumberOfElements(mxx);

    if (mxIsSingle(mxx)) {
        float *y = (float *)mxGetData(mxy);
        const float *x = (const float *)mxGetData(mxx);
        scopy(&n, x, &incx, y, &incy);

    } else if (mxIsDouble(mxx)) {
        double *y = (double *)mxGetData(mxy);
        const double *x = (const double *)mxGetData(mxx);
        dcopy(&n, x, &incx, y, &incy);

    } else {
        mexErrMsgTxt("not implemented");
    }

    return;
} /* mx_copy() */


void
mx_axpy(const double a, const mxArray *mxx, mxArray *mxy)
{
    const ptrdiff_t incx = 1;
    const ptrdiff_t incy = 1;
    const ptrdiff_t n = (const ptrdiff_t)mxGetNumberOfElements(mxx);

    if (mxIsSingle(mxx)) {
        float *y = (float *)mxGetData(mxy);
        const float *x = (const float *)mxGetData(mxx);
        const float af = (const float)a;
        saxpy(&n, &af, x, &incx, y, &incy);

    } else if (mxIsDouble(mxx)) {
        double *y = (double *)mxGetData(mxy);
        const double *x = (const double *)mxGetData(mxx);
        daxpy(&n, &a, x, &incx, y, &incy);

    } else {
        mexErrMsgTxt("not implemented");
    }

    return;
} /* mx_axpy() */


double
mx_dot(const mxArray *mxx, const mxArray *mxy)
{
    double p = -1.0;

    const ptrdiff_t incx = 1;
    const ptrdiff_t incy = 1;
    const ptrdiff_t n = (const ptrdiff_t)mxGetNumberOfElements(mxx);

    if (mxIsSingle(mxx)) {
        const float *x = (const float *)mxGetData(mxx);
        const float *y = (const float *)mxGetData(mxy);
        p = (double)sdot(&n, x, &incx, y, &incy);

    } else if (mxIsDouble(mxx)) {
        const double *x = (const double *)mxGetData(mxx);
        const double *y = (const double *)mxGetData(mxy);
        p = ddot(&n, x, &incx, y, &incy);

    } else {
        mexErrMsgTxt("not implemented");
    }

    return p;
} /* mx_dot() */


double
mx_nrm2(const mxArray *mxx)
{
    double p = -1.0;

    const ptrdiff_t incx = 1;
    const ptrdiff_t n = (const ptrdiff_t)mxGetNumberOfElements(mxx);

    if (mxIsSingle(mxx)) {
        const float *x = (const float *)mxGetData(mxx);
        p = (double)snrm2(&n, x, &incx);

    } else if (mxIsDouble(mxx)) {
        const double *x = (const double *)mxGetData(mxx);
        p = dnrm2(&n, x, &incx);

    } else {
        mexErrMsgTxt("not implemented");
    }

    return p;
} /* mx_nrm2() */


double
mx_asum(const mxArray *mxx)
{
    double p = -1.0;

    const ptrdiff_t incx = 1;
    const ptrdiff_t n = (const ptrdiff_t)mxGetNumberOfElements(mxx);

    if (mxIsSingle(mxx)) {
        const float *x = (const float *)mxGetData(mxx);
        p = (double)sasum(&n, x, &incx);

    } else if (mxIsDouble(mxx)) {
        const double *x = (const double *)mxGetData(mxx);
        p = dasum(&n, x, &incx);

    } else {
        mexErrMsgTxt("not implemented");
    }

    return p;
} /* mx_asum() */
