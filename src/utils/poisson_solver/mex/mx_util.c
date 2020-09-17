#include <inttypes.h>
#include <omp.h>
#include <string.h>
#include "matrix.h"
#include "mx_util.h"


mxArray *
mx_pad_boundary(const mxArray *mxx)
{
    const size_t *sz = (const size_t *)mxGetDimensions(mxx);
    const size_t szp[3] = {sz[0]+2, sz[1]+2, sz[2]+2};

    mxArray *mxxp = mxCreateNumericArray(3, szp, mxGetClassID(mxx), mxREAL);

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t nxp = szp[0];
    const size_t nyp = szp[1];
    const size_t nxnyp = nxp*nyp;

    if (mxIsSingle(mxx)) {
        float *x = (float *)mxGetData(mxx);
        float *xp = (float *)mxGetData(mxxp);

        #pragma omp parallel for schedule(static) collapse(3) \
            if(nxny*nz > 32*32*32)
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    xp[(i+1) + nxp*(j+1) + nxnyp*(k+1)] = x[i + nx*j + nxny*k];
                }
            }
        }

    } else if (mxIsDouble(mxx)) {
        double *x = (double *)mxGetData(mxx);
        double *xp = (double *)mxGetData(mxxp);

        #pragma omp parallel for schedule(static) collapse(3) \
            if(nxny*nz > 32*32*32)
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    xp[(i+1) + nxp*(j+1) + nxnyp*(k+1)] = x[i + nx*j + nxny*k];
                }
            }
        }

    } else if (mxIsLogical(mxx)) {
        uint8_t *x = (uint8_t *)mxGetData(mxx);
        uint8_t *xp = (uint8_t *)mxGetData(mxxp);

        #pragma omp parallel for schedule(static) collapse(3) \
            if(nxny*nz > 32*32*32)
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    xp[(i+1) + nxp*(j+1) + nxnyp*(k+1)] = x[i + nx*j + nxny*k];
                }
            }
        }
    }

    return mxxp;
}


mxArray *
mx_unpad_boundary(const mxArray *mxxp)
{
    const size_t *szp = (const size_t *)mxGetDimensions(mxxp);
    const size_t sz[3] = {szp[0]-2, szp[1]-2, szp[2]-2};

    mxArray *mxx = mxCreateNumericArray(3, sz, mxGetClassID(mxxp), mxREAL);

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t nxp = szp[0];
    const size_t nyp = szp[1];
    const size_t nxnyp = nxp*nyp;

    if (mxIsSingle(mxxp)) {
        float *x = (float *)mxGetData(mxx);
        float *xp = (float *)mxGetData(mxxp);

        #pragma omp parallel for schedule(static) collapse(3) \
            if(nxny*nz > 32*32*32)
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    x[i + nx*j + nxny*k] = xp[(i+1) + nxp*(j+1) + nxnyp*(k+1)];
                }
            }
        }

    } else if (mxIsDouble(mxxp)) {
        double *x = (double *)mxGetData(mxx);
        double *xp = (double *)mxGetData(mxxp);

        #pragma omp parallel for schedule(static) collapse(3) \
            if(nxny*nz > 32*32*32)
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    x[i + nx*j + nxny*k] = xp[(i+1) + nxp*(j+1) + nxnyp*(k+1)];
                }
            }
        }

    } else if (mxIsLogical(mxxp)) {
        uint8_t *x = (uint8_t *)mxGetData(mxx);
        uint8_t *xp = (uint8_t *)mxGetData(mxxp);

        #pragma omp parallel for schedule(static) collapse(3) \
            if(nxny*nz > 32*32*32)
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    x[i + nx*j + nxny*k] = xp[(i+1) + nxp*(j+1) + nxnyp*(k+1)];
                }
            }
        }
    }

    return mxx;
}


void
mx_zero(mxArray *mxx)
{
    const size_t n = mxGetNumberOfElements(mxx);
    const size_t b = mxGetElementSize(mxx);
    memset(mxGetData(mxx), 0, n*b);
    return;
}
