#include <inttypes.h>
#include <omp.h>
#include "mex.h"


void gradfm_adjf(float *du,
                const float *x, const float *y, const float *z,
                const uint8_t *G, const double *h, const size_t *sz);

void gradfm_adjd(double *du,
                const double *x, const double *y, const double *z,
                const uint8_t *G, const double *h, const size_t *sz);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 6) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: gradfm_adj_mex(du, x, y, z, G, h);");
        return;
    }

    const uint8_t *G = (const uint8_t *)mxGetData(prhs[4]);
    const double *h = (const double *)mxGetData(prhs[5]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *du = (float *)mxGetData(prhs[0]);
        const float *x = (const float *)mxGetData(prhs[1]);
        const float *y = (const float *)mxGetData(prhs[2]);
        const float *z = (const float *)mxGetData(prhs[3]);

        gradfm_adjf(du, x, y, z, G, h, sz);

    } else {
        double *du = (double *)mxGetData(prhs[0]);
        const double *x = (const double *)mxGetData(prhs[1]);
        const double *y = (const double *)mxGetData(prhs[2]);
        const double *z = (const double *)mxGetData(prhs[3]);

        gradfm_adjd(du, x, y, z, G, h, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}


void
gradfm_adjf(float *du,
            const float *x, const float *y, const float *z,
            const uint8_t *G, const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    float dx, dy, dz;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;
    const size_t nxnynz = nx*ny*nz;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const float hx = (float)(-1.0/h[0]);
    const float hy = (float)(-1.0/h[1]);
    const float hz = (float)(-1.0/h[2]);

    #pragma omp parallel for private(i,j,k,l) schedule(static) \
        if(nxnynz > 16*16*16)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = j + k;
            for(i = 0; i < nx; ++i, ++l) {
                if (G[l]) {
                    dz =
                        (k > 0) && G[l-nxny] ? hz*(z[l]-z[l-nxny]) :
                        (k < NZ) && G[l+nxny] ? hz*(z[l+nxny]-z[l]) :
                        0.0f;

                    dy =
                        (j > 0) && G[l-nx] ? hy*(y[l]-y[l-nx]) :
                        (j < NY) && G[l+nx] ? hy*(y[l+nx]-y[l]) :
                        0.0f;

                    dx =
                        (i > 0) && G[l-1] ? hx*(x[l]-x[l-1]) :
                        (i < NX) && G[l+1] ? hx*(x[l+1]-x[l]) :
                        0.0f;

                    du[l] = dx + dy + dz;
                }
            }
        }
    }

    return;
}


void
gradfm_adjd(double *du,
            const double *x, const double *y, const double *z,
            const uint8_t *G, const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    double dx, dy, dz;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;
    const size_t nxnynz = nx*ny*nz;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const double hx = -1.0/h[0];
    const double hy = -1.0/h[1];
    const double hz = -1.0/h[2];

    #pragma omp parallel for private(i,j,k,l) schedule(static) \
        if(nxnynz > 16*16*16)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = j + k;
            for(i = 0; i < nx; ++i, ++l) {
                if (G[l]) {
                    dz =
                        (k > 0) && G[l-nxny] ? hz*(z[l]-z[l-nxny]) :
                        (k < NZ) && G[l+nxny] ? hz*(z[l+nxny]-z[l]) :
                        0.0;

                    dy =
                        (j > 0) && G[l-nx] ? hy*(y[l]-y[l-nx]) :
                        (j < NY) && G[l+nx] ? hy*(y[l+nx]-y[l]) :
                        0.0;

                    dx =
                        (i > 0) && G[l-1] ? hx*(x[l]-x[l-1]) :
                        (i < NX) && G[l+1] ? hx*(x[l+1]-x[l]) :
                        0.0;

                    du[l] = dx + dy + dz;
                }
            }
        }
    }

    return;
}
