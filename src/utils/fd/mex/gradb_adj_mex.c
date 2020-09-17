#include <inttypes.h>
#include <omp.h>
#include "mex.h"


void gradb_adjf(float *du,
                const float *x, const float *y, const float *z,
                const double *h, const size_t *sz);

void gradb_adjd(double *du,
                const double *x, const double *y, const double *z,
                const double *h, const size_t *sz);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 5) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: gradb_adj_mex(du, x, y, z, h);");
        return;
    }

    const double *h = (const double *)mxGetData(prhs[4]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *du = (float *)mxGetData(prhs[0]);
        const float *x = (const float *)mxGetData(prhs[1]);
        const float *y = (const float *)mxGetData(prhs[2]);
        const float *z = (const float *)mxGetData(prhs[3]);

        gradb_adjf(du, x, y, z, h, sz);

    } else {
        double *du = (double *)mxGetData(prhs[0]);
        const double *x = (const double *)mxGetData(prhs[1]);
        const double *y = (const double *)mxGetData(prhs[2]);
        const double *z = (const double *)mxGetData(prhs[3]);

        gradb_adjd(du, x, y, z, h, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}


void
gradb_adjf(float *du,
           const float *x, const float *y, const float *z,
           const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const float hx = (float)(-1.0/h[0]);
    const float hy = (float)(-1.0/h[1]);
    const float hz = (float)(-1.0/h[2]);

#pragma omp parallel private(i,j,k,l) if (nxny*nz > 16*16*16)
{
    #pragma omp for schedule(static)
    for(k = 0; k < NZ; k += nxny) {
        for(j = 0; j < NY; j += nx) {
            l = j + k;
            for(i = 0; i < NX; ++i, ++l) {
                du[l] =
                    hx*(x[l+1]-x[l]) +
                    hy*(y[l+nx]-y[l]) +
                    hz*(z[l+nxny]-z[l]);
            }

            /* i = nx-1 */
            l = NX + j + k;
            du[l] =
                hx*(x[l]-x[l-1]) +
                hy*(y[l+nx]-y[l]) +
                hz*(z[l+nxny]-z[l]);

        }

        /* j = ny-1 */
        l = NY + k;
        for(i = 0; i < NX; ++i, ++l) {
            du[l] =
                hx*(x[l+1]-x[l]) +
                hy*(y[l]-y[l-nx]) +
                hz*(z[l+nxny]-z[l]);
        }

        /* i = nx-1, j = ny-1 */
        l = NX + NY + k;
        du[l] =
            hx*(x[l]-x[l-1]) +
            hy*(y[l]-y[l-nx]) +
            hz*(z[l+nxny]-z[l]);

    }

    /* k = nz-1 */
    #pragma omp for schedule(static) collapse(2)
    for(j = 0; j < NY; j += nx) {
        for(i = 0; i < NX; ++i) {
            l = i + j + NZ;
            du[l] =
                hx*(x[l+1]-x[l]) +
                hy*(y[l+nx]-y[l]) +
                hz*(z[l]-z[l-nxny]);
        }
    }

    /* j = ny-1, k = nz-1 */
    #pragma omp for schedule(static)
    for(i = 0; i < NX; ++i) {
        l = i + NY + NZ;
        du[l] =
            hx*(x[l+1]-x[l]) +
            hy*(y[l]-y[l-nx]) +
            hz*(z[l]-z[l-nxny]);
    }

    /* i = nx-1, k = nz-1 */
    #pragma omp for schedule(static)
    for(j = 0; j < NY; j += nx) {
        l = NX + j + NZ;
        du[l] =
            hx*(x[l]-x[l-1]) +
            hy*(y[l+nx]-y[l]) +
            hz*(z[l]-z[l-nxny]);
    }

} /* omp parallel */

    /* i = nx-1, j = ny-1, k = nz-1 */
    l = NX + NY + NZ;
    du[l] =
        hx*(x[l]-x[l-1]) +
        hy*(y[l]-y[l-nx]) +
        hz*(z[l]-z[l-nxny]);

    return;
}


void
gradb_adjd(double *du,
           const double *x, const double *y, const double *z,
           const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const double hx = -1.0/h[0];
    const double hy = -1.0/h[1];
    const double hz = -1.0/h[2];

#pragma omp parallel private(i,j,k,l) if (nxny*nz > 16*16*16)
{
    #pragma omp for schedule(static)
    for(k = 0; k < NZ; k += nxny) {
        for(j = 0; j < NY; j += nx) {
            l = j + k;
            for(i = 0; i < NX; ++i, ++l) {
                du[l] =
                    hx*(x[l+1]-x[l]) +
                    hy*(y[l+nx]-y[l]) +
                    hz*(z[l+nxny]-z[l]);
            }

            /* i = nx-1 */
            l = NX + j + k;
            du[l] =
                hx*(x[l]-x[l-1]) +
                hy*(y[l+nx]-y[l]) +
                hz*(z[l+nxny]-z[l]);

        }

        /* j = ny-1 */
        l = NY + k;
        for(i = 0; i < NX; ++i, ++l) {
            du[l] =
                hx*(x[l+1]-x[l]) +
                hy*(y[l]-y[l-nx]) +
                hz*(z[l+nxny]-z[l]);
        }

        /* i = nx-1, j = ny-1 */
        l = NX + NY + k;
        du[l] =
            hx*(x[l]-x[l-1]) +
            hy*(y[l]-y[l-nx]) +
            hz*(z[l+nxny]-z[l]);

    }

    /* k = nz-1 */
    #pragma omp for schedule(static) collapse(2)
    for(j = 0; j < NY; j += nx) {
        for(i = 0; i < NX; ++i) {
            l = i + j + NZ;
            du[l] =
                hx*(x[l+1]-x[l]) +
                hy*(y[l+nx]-y[l]) +
                hz*(z[l]-z[l-nxny]);
        }
    }

    /* j = ny-1, k = nz-1 */
    #pragma omp for schedule(static)
    for(i = 0; i < NX; ++i) {
        l = i + NY + NZ;
        du[l] =
            hx*(x[l+1]-x[l]) +
            hy*(y[l]-y[l-nx]) +
            hz*(z[l]-z[l-nxny]);
    }

    /* i = nx-1, k = nz-1 */
    #pragma omp for schedule(static)
    for(j = 0; j < NY; j += nx) {
        l = NX + j + NZ;
        du[l] =
            hx*(x[l]-x[l-1]) +
            hy*(y[l+nx]-y[l]) +
            hz*(z[l]-z[l-nxny]);
    }

} /* omp parallel */

    /* i = nx-1, j = ny-1, k = nz-1 */
    l = NX + NY + NZ;
    du[l] =
        hx*(x[l]-x[l-1]) +
        hy*(y[l]-y[l-nx]) +
        hz*(z[l]-z[l-nxny]);

    return;
}
