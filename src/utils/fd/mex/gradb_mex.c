#include <inttypes.h>
#include <omp.h>
#include "mex.h"


void gradbf(float *dx, float *dy, float *dz,
            const float *u, const double *h, const size_t *sz);

void gradbd(double *dx, double *dy, double *dz,
            const double *u, const double *h, const size_t *sz);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 5) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: gradb_mex(dx, dy, dz, u, h);");
        return;
    }

    const double *h = (const double *)mxGetData(prhs[4]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *dx = (float *)mxGetData(prhs[0]);
        float *dy = (float *)mxGetData(prhs[1]);
        float *dz = (float *)mxGetData(prhs[2]);
        const float *u = (const float *)mxGetData(prhs[3]);

        gradbf(dx, dy, dz, u, h, sz);

    } else {
        double *dx = (double *)mxGetData(prhs[0]);
        double *dy = (double *)mxGetData(prhs[1]);
        double *dz = (double *)mxGetData(prhs[2]);
        const double *u = (const double *)mxGetData(prhs[3]);

        gradbd(dx, dy, dz, u, h, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}


void
gradbf(float *dx, float *dy, float *dz,
       const float *u, const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;
    const size_t nxnynz = nx*ny*nz;

    const float hx = (float)(1.0/h[0]);
    const float hy = (float)(1.0/h[1]);
    const float hz = (float)(1.0/h[2]);

    /* i = 0, j = 0, k = 0 */
    l = 0;
    dx[l] = hx*(u[l+1]-u[l]);
    dy[l] = hy*(u[l+nx]-u[l]);
    dz[l] = hz*(u[l+nxny]-u[l]);

#pragma omp parallel private(i,j,k,l) if (nxny*nz > 16*16*16)
{
    /* i = 0, k = 0 */
    #pragma omp for schedule(static)
    for(l = nx; l < nxny; l += nx) {
        dx[l] = hx*(u[l+1]-u[l]);
        dy[l] = hy*(u[l]-u[l-nx]);
        dz[l] = hz*(u[l+nxny]-u[l]);
    }

    /* j = 0, k = 0 */
    #pragma omp for schedule(static)
    for(l = 1; l < nx; ++l) {
        dx[l] = hx*(u[l]-u[l-1]);
        dy[l] = hy*(u[l+nx]-u[l]);
        dz[l] = hz*(u[l+nxny]-u[l]);
    }

    /* k = 0 */
    #pragma omp for schedule(static) collapse(2)
    for(j = nx; j < nxny; j += nx) {
        for(i = 1; i < nx; ++i) {
            l = i + j;
            dx[l] = hx*(u[l]-u[l-1]);
            dy[l] = hy*(u[l]-u[l-nx]);
            dz[l] = hz*(u[l+nxny]-u[l]);
        }
    }

    /* interior loop */
    #pragma omp for schedule(static)
    for(k = nxny; k < nxnynz; k += nxny) {
        /* i = 0, j = 0 */
        l = k;
        dx[l] = hx*(u[l+1]-u[l]);
        dy[l] = hy*(u[l+nx]-u[l]);
        dz[l] = hz*(u[l]-u[l-nxny]);

        /* j = 0 */
        l = 1 + k;
        for(i = 1; i < nx; ++i, ++l) {
            dx[l] = hx*(u[l]-u[l-1]);
            dy[l] = hy*(u[l+nx]-u[l]);
            dz[l] = hz*(u[l]-u[l-nxny]);
        }

        for(j = nx; j < nxny; j += nx) {
            /* i = 0 */
            l = j + k;
            dx[l] = hx*(u[l+1]-u[l]);
            dy[l] = hy*(u[l]-u[l-nx]);
            dz[l] = hz*(u[l]-u[l-nxny]);

            l = 1 + j + k;
            for(i = 1; i < nx; ++i, ++l) {
                dx[l] = hx*(u[l]-u[l-1]);
                dy[l] = hy*(u[l]-u[l-nx]);
                dz[l] = hz*(u[l]-u[l-nxny]);
            }
        }
    }

} /* omp parallel */

    return;
}


void
gradbd(double *dx, double *dy, double *dz,
       const double *u, const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;
    const size_t nxnynz = nx*ny*nz;

    const double hx = 1.0/h[0];
    const double hy = 1.0/h[1];
    const double hz = 1.0/h[2];

    /* i = 0, j = 0, k = 0 */
    l = 0;
    dx[l] = hx*(u[l+1]-u[l]);
    dy[l] = hy*(u[l+nx]-u[l]);
    dz[l] = hz*(u[l+nxny]-u[l]);

#pragma omp parallel private(i,j,k,l) if (nxny*nz > 16*16*16)
{
    /* i = 0, k = 0 */
    #pragma omp for schedule(static)
    for(l = nx; l < nxny; l += nx) {
        dx[l] = hx*(u[l+1]-u[l]);
        dy[l] = hy*(u[l]-u[l-nx]);
        dz[l] = hz*(u[l+nxny]-u[l]);
    }

    /* j = 0, k = 0 */
    #pragma omp for schedule(static)
    for(l = 1; l < nx; ++l) {
        dx[l] = hx*(u[l]-u[l-1]);
        dy[l] = hy*(u[l+nx]-u[l]);
        dz[l] = hz*(u[l+nxny]-u[l]);
    }

    /* k = 0 */
    #pragma omp for schedule(static) collapse(2)
    for(j = nx; j < nxny; j += nx) {
        for(i = 1; i < nx; ++i) {
            l = i + j;
            dx[l] = hx*(u[l]-u[l-1]);
            dy[l] = hy*(u[l]-u[l-nx]);
            dz[l] = hz*(u[l+nxny]-u[l]);
        }
    }

    /* interior loop */
    #pragma omp for schedule(static)
    for(k = nxny; k < nxnynz; k += nxny) {
        /* i = 0, j = 0 */
        l = k;
        dx[l] = hx*(u[l+1]-u[l]);
        dy[l] = hy*(u[l+nx]-u[l]);
        dz[l] = hz*(u[l]-u[l-nxny]);

        /* j = 0 */
        l = 1 + k;
        for(i = 1; i < nx; ++i, ++l) {
            dx[l] = hx*(u[l]-u[l-1]);
            dy[l] = hy*(u[l+nx]-u[l]);
            dz[l] = hz*(u[l]-u[l-nxny]);
        }

        for(j = nx; j < nxny; j += nx) {
            /* i = 0 */
            l = j + k;
            dx[l] = hx*(u[l+1]-u[l]);
            dy[l] = hy*(u[l]-u[l-nx]);
            dz[l] = hz*(u[l]-u[l-nxny]);

            l = 1 + j + k;
            for(i = 1; i < nx; ++i, ++l) {
                dx[l] = hx*(u[l]-u[l-1]);
                dy[l] = hy*(u[l]-u[l-nx]);
                dz[l] = hz*(u[l]-u[l-nxny]);
            }
        }
    }

} /* omp parallel */

    return;
}
