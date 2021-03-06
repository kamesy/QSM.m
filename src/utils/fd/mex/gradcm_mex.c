#include <inttypes.h>
#include <omp.h>
#include "mex.h"


void gradcmf(float *dx, float *dy, float *dz,
             const float *u, const uint8_t *G,
             const double *h, const size_t *sz);

void gradcmd(double *dx, double *dy, double *dz,
             const double *u, const uint8_t *G,
             const double *h, const size_t *sz);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 6) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: gradcm_mex(dx, dy, dz, u, G, h);");
        return;
    }

    const uint8_t *G = (const uint8_t *)mxGetData(prhs[4]);
    const double *h = (const double *)mxGetData(prhs[5]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *dx = (float *)mxGetData(prhs[0]);
        float *dy = (float *)mxGetData(prhs[1]);
        float *dz = (float *)mxGetData(prhs[2]);
        const float *u = (const float *)mxGetData(prhs[3]);

        gradcmf(dx, dy, dz, u, G, h, sz);

    } else {
        double *dx = (double *)mxGetData(prhs[0]);
        double *dy = (double *)mxGetData(prhs[1]);
        double *dz = (double *)mxGetData(prhs[2]);
        const double *u = (const double *)mxGetData(prhs[3]);

        gradcmd(dx, dy, dz, u, G, h, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}


void
gradcmf(float *dx, float *dy, float *dz,
        const float *u, const uint8_t *G,
        const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;
    const size_t nxnynz = nx*ny*nz;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const float hx = (float)(1.0/h[0]);
    const float hy = (float)(1.0/h[1]);
    const float hz = (float)(1.0/h[2]);

    const float hx2 = (float)(0.5/h[0]);
    const float hy2 = (float)(0.5/h[1]);
    const float hz2 = (float)(0.5/h[2]);

    #pragma omp parallel for private(i,j,k,l) schedule(static) \
        if(nxnynz > 16*16*16)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = j + k;
            for(i = 0; i < nx; ++i, ++l) {
                if (G[l]) {
                    dx[l] =
                        (i > 0) && (i < NX) && G[l-1] && G[l+1] ?
                            hx2*(u[l+1]-u[l-1]) :
                        (i < NX) && G[l+1] ?
                            hx*(u[l+1]-u[l]) :
                        (i > 0) && G[l-1] ?
                            hx*(u[l]-u[l-1]) :
                        0.0f;

                    dy[l] =
                        (j > 0) && (j < NY) && G[l-nx] && G[l+nx] ?
                            hy2*(u[l+nx]-u[l-nx]) :
                        (j < NY) && G[l+nx] ?
                            hy*(u[l+nx]-u[l]) :
                        (j > 0) && G[l-nx] ?
                            hy*(u[l]-u[l-nx]) :
                        0.0f;

                    dz[l] =
                        (k > 0) && (k < NZ) && G[l-nxny] && G[l+nxny] ?
                            hz2*(u[l+nxny]-u[l-nxny]) :
                        (k < NZ) && G[l+nxny] ?
                            hz*(u[l+nxny]-u[l]) :
                        (k > 0) && G[l-nxny] ?
                            hz*(u[l]-u[l-nxny]) :
                        0.0f;
                }
            }
        }
    }

    return;
}


void
gradcmd(double *dx, double *dy, double *dz,
        const double *u, const uint8_t *G,
        const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;
    const size_t nxnynz = nx*ny*nz;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const double hx = 1.0/h[0];
    const double hy = 1.0/h[1];
    const double hz = 1.0/h[2];

    const double hx2 = 0.5/h[0];
    const double hy2 = 0.5/h[1];
    const double hz2 = 0.5/h[2];

    #pragma omp parallel for private(i,j,k,l) schedule(static) \
        if(nxnynz > 16*16*16)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = j + k;
            for(i = 0; i < nx; ++i, ++l) {
                if (G[l]) {
                    dx[l] =
                        (i > 0) && (i < NX) && G[l-1] && G[l+1] ?
                            hx2*(u[l+1]-u[l-1]) :
                        (i < NX) && G[l+1] ?
                            hx*(u[l+1]-u[l]) :
                        (i > 0) && G[l-1] ?
                            hx*(u[l]-u[l-1]) :
                        0.0;

                    dy[l] =
                        (j > 0) && (j < NY) && G[l-nx] && G[l+nx] ?
                            hy2*(u[l+nx]-u[l-nx]) :
                        (j < NY) && G[l+nx] ?
                            hy*(u[l+nx]-u[l]) :
                        (j > 0) && G[l-nx] ?
                            hy*(u[l]-u[l-nx]) :
                        0.0;

                    dz[l] =
                        (k > 0) && (k < NZ) && G[l-nxny] && G[l+nxny] ?
                            hz2*(u[l+nxny]-u[l-nxny]) :
                        (k < NZ) && G[l+nxny] ?
                            hz*(u[l+nxny]-u[l]) :
                        (k > 0) && G[l-nxny] ?
                            hz*(u[l]-u[l-nxny]) :
                        0.0;
                }
            }
        }
    }

    return;
}
