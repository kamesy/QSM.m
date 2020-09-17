#include <inttypes.h>
#include <omp.h>
#include "mex.h"
#include "lapmg_mex.h"


void lapmgf(float *du,
            const float *u, const uint8_t *G,
            const double *h, const size_t *sz);

void lapmgd(double *du,
            const double *u, const uint8_t *G,
            const double *h, const size_t *sz);


#ifdef LAPMG_MEX
void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 4) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: lapmg_mex(d2u, u, G, h);");
    }

    const uint8_t *G = (const uint8_t *)mxGetData(prhs[2]);
    const double *h = (const double *)mxGetData(prhs[3]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *du = (float *)mxGetData(prhs[0]);
        const float *u = (const float *)mxGetData(prhs[1]);
        lapmgf(du, u, G, h, sz);

    } else {
        double *du = (double *)mxGetData(prhs[0]);
        const double *u = (const double *)mxGetData(prhs[1]);
        lapmgd(du, u, G, h, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}
#endif


void
mx_lapmg(mxArray *mxdu,
         const mxArray *mxu, const mxArray *mxG, const mxArray *mxh)
{
    const uint8_t *G = (const uint8_t *)mxGetData(mxG);
    const double *h = (const double *)mxGetData(mxh);

    const size_t *sz = (const size_t *)mxGetDimensions(mxdu);

    if (mxIsSingle(mxdu)) {
        float *du = (float *)mxGetData(mxdu);
        const float *u = (const float *)mxGetData(mxu);
        lapmgf(du, u, G, h, sz);

    } else {
        double *du = (double *)mxGetData(mxdu);
        const double *u = (const double *)mxGetData(mxu);
        lapmgd(du, u, G, h, sz);
    }

    return;
}


void
lapmgf(float *du,
       const float *u, const uint8_t *G, const double *h, const size_t *sz)
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

    const float hx = (float)(1.0/(h[0]*h[0]));
    const float hy = (float)(1.0/(h[1]*h[1]));
    const float hz = (float)(1.0/(h[2]*h[2]));
    const float hh = (float)(-2.0*(hx+hy+hz));

    #pragma omp parallel for private(i,j,k,l) schedule(static) \
        if(nxny*nz > 32*32*32)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                if (G[l]) {
                    du[l] =
                        hh*u[l] +
                        hx*(u[l-1] + u[l+1]) +
                        hy*(u[l-nx] + u[l+nx]) +
                        hz*(u[l-nxny] + u[l+nxny]);
                }
            }
        }
    }

    return;
}


void
lapmgd(double *du,
       const double *u, const uint8_t *G, const double *h, const size_t *sz)
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

    const double hx = 1.0/(h[0]*h[0]);
    const double hy = 1.0/(h[1]*h[1]);
    const double hz = 1.0/(h[2]*h[2]);
    const double hh = -2.0*(hx+hy+hz);

    #pragma omp parallel for private(i,j,k,l) schedule(static) \
        if(nxny*nz > 32*32*32)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                if (G[l]) {
                    du[l] =
                        hh*u[l] +
                        hx*(u[l-1] + u[l+1]) +
                        hy*(u[l-nx] + u[l+nx]) +
                        hz*(u[l-nxny] + u[l+nxny]);
                }
            }
        }
    }

    return;
}
