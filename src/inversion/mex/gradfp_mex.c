#include <inttypes.h>
#include <omp.h>
#include "mex.h"


void gradfpf(float *dx, float *dy, float *dz,
             const float *u, const double *h, const size_t *sz);

void gradfpd(double *dx, double *dy, double *dz,
             const double *u, const double *h, const size_t *sz);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 5) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: gradfp_mex(dx, dy, dz, u, h);");
        return;
    }

    const double *h = (const double *)mxGetData(prhs[4]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *dx = (float *)mxGetData(prhs[0]);
        float *dy = (float *)mxGetData(prhs[1]);
        float *dz = (float *)mxGetData(prhs[2]);
        const float *u = (const float *)mxGetData(prhs[3]);

        gradfpf(dx, dy, dz, u, h, sz);

    } else {
        double *dx = (double *)mxGetData(prhs[0]);
        double *dy = (double *)mxGetData(prhs[1]);
        double *dz = (double *)mxGetData(prhs[2]);
        const double *u = (const double *)mxGetData(prhs[3]);

        gradfpd(dx, dy, dz, u, h, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}


void
gradfpf(float *dx, float *dy, float *dz,
        const float *u, const double *h, const size_t *sz)
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

    const float hx = (float)(1.0/h[0]);
    const float hy = (float)(1.0/h[1]);
    const float hz = (float)(1.0/h[2]);

#pragma omp parallel private(i,j,k,l)
{
    #pragma omp for schedule(static)
    for(k = 0; k < NZ; k += nxny) {
        for(j = 0; j < NY; j += nx) {
            l = j + k;
            for(i = 0; i < NX; ++i, ++l) {
                dx[l] = hx*(u[l+1]-u[l]);
                dy[l] = hy*(u[l+nx]-u[l]);
                dz[l] = hz*(u[l+nxny]-u[l]);
            }

            /* i = nx-1 */
            l = NX + j + k;
            dx[l] = hx*(u[l-NX]-u[l]);
            dy[l] = hy*(u[l+nx]-u[l]);
            dz[l] = hz*(u[l+nxny]-u[l]);

        }

        /* j = ny-1 */
        l = NY + k;
        for(i = 0; i < NX; ++i, ++l) {
            dy[l] = hy*(u[l-NY]-u[l]);
            dx[l] = hx*(u[l+1]-u[l]);
            dz[l] = hz*(u[l+nxny]-u[l]);
        }

        /* i = nx-1, j = ny-1 */
        l = NX + NY + k;
        dy[l] = hy*(u[l-NY]-u[l]);
        dx[l] = hx*(u[l-NX]-u[l]);
        dz[l] = hz*(u[l+nxny]-u[l]);

    }

    /* k = nz-1 */
    #pragma omp for schedule(static) collapse(2)
    for(j = 0; j < NY; j += nx) {
        for(i = 0; i < NX; ++i) {
            l = i + j + NZ;
            dz[l] = hz*(u[l-NZ]-u[l]);
            dx[l] = hx*(u[l+1]-u[l]);
            dy[l] = hy*(u[l+nx]-u[l]);
        }
    }

    /* j = ny-1, k = nz-1 */
    #pragma omp for schedule(static)
    for(i = 0; i < NX; ++i) {
        l = i + NY + NZ;
        dz[l] = hz*(u[l-NZ]-u[l]);
        dy[l] = hy*(u[l-NY]-u[l]);
        dx[l] = hx*(u[l+1]-u[l]);
    }

    /* i = nx-1, k = nz-1 */
    #pragma omp for schedule(static)
    for(j = 0; j < NY; j += nx) {
        l = j + NX + NZ;
        dz[l] = hz*(u[l-NZ]-u[l]);
        dx[l] = hx*(u[l-NX]-u[l]);
        dy[l] = hy*(u[l+nx]-u[l]);
    }

} /* omp parallel */

    /* i = nx-1, j = ny-1, k = nz-1 */
    l = NX + NY + NZ;
    dz[l] = hz*(u[l-NZ]-u[l]);
    dy[l] = hy*(u[l-NY]-u[l]);
    dx[l] = hx*(u[l-NX]-u[l]);

    return;
}


void
gradfpd(double *dx, double *dy, double *dz,
        const double *u, const double *h, const size_t *sz)
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

    const double hx = 1.0/h[0];
    const double hy = 1.0/h[1];
    const double hz = 1.0/h[2];


#pragma omp parallel private(i,j,k,l)
{
    #pragma omp for schedule(static)
    for(k = 0; k < NZ; k += nxny) {
        for(j = 0; j < NY; j += nx) {
            l = j + k;
            for(i = 0; i < NX; ++i, ++l) {
                dx[l] = hx*(u[l+1]-u[l]);
                dy[l] = hy*(u[l+nx]-u[l]);
                dz[l] = hz*(u[l+nxny]-u[l]);
            }

            /* i = nx-1 */
            l = NX + j + k;
            dx[l] = hx*(u[l-NX]-u[l]);
            dy[l] = hy*(u[l+nx]-u[l]);
            dz[l] = hz*(u[l+nxny]-u[l]);

        }

        /* j = ny-1 */
        l = NY + k;
        for(i = 0; i < NX; ++i, ++l) {
            dy[l] = hy*(u[l-NY]-u[l]);
            dx[l] = hx*(u[l+1]-u[l]);
            dz[l] = hz*(u[l+nxny]-u[l]);
        }

        /* i = nx-1, j = ny-1 */
        l = NX + NY + k;
        dy[l] = hy*(u[l-NY]-u[l]);
        dx[l] = hx*(u[l-NX]-u[l]);
        dz[l] = hz*(u[l+nxny]-u[l]);

    }

    /* k = nz-1 */
    #pragma omp for schedule(static) collapse(2)
    for(j = 0; j < NY; j += nx) {
        for(i = 0; i < NX; ++i) {
            l = i + j + NZ;
            dz[l] = hz*(u[l-NZ]-u[l]);
            dx[l] = hx*(u[l+1]-u[l]);
            dy[l] = hy*(u[l+nx]-u[l]);
        }
    }

    /* j = ny-1, k = nz-1 */
    #pragma omp for schedule(static)
    for(i = 0; i < NX; ++i) {
        l = i + NY + NZ;
        dz[l] = hz*(u[l-NZ]-u[l]);
        dy[l] = hy*(u[l-NY]-u[l]);
        dx[l] = hx*(u[l+1]-u[l]);
    }

    /* i = nx-1, k = nz-1 */
    #pragma omp for schedule(static)
    for(j = 0; j < NY; j += nx) {
        l = j + NX + NZ;
        dz[l] = hz*(u[l-NZ]-u[l]);
        dx[l] = hx*(u[l-NX]-u[l]);
        dy[l] = hy*(u[l+nx]-u[l]);
    }

} /* omp parallel */

    /* i = nx-1, j = ny-1, k = nz-1 */
    l = NX + NY + NZ;
    dz[l] = hz*(u[l-NZ]-u[l]);
    dy[l] = hy*(u[l-NY]-u[l]);
    dx[l] = hx*(u[l-NX]-u[l]);

    return;
}
