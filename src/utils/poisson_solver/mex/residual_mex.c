#include <inttypes.h>
#include <omp.h>
#include "mex.h"
#include "residual_mex.h"


void residualf(float *r,
               const float *f, const float *x, const uint8_t *G,
               const double *h, const size_t *sz);

void residuald(double *r,
               const double *f, const double *x, const uint8_t *G,
               const double *h, const size_t *sz);


#ifdef RESIDUAL_MEX
void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 5) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: residual_mex(r, f, x, G, h);");
    }

    const uint8_t *G = (const uint8_t *)mxGetData(prhs[3]);
    const double *h = (const double *)mxGetData(prhs[4]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *r = (float *)mxGetData(prhs[0]);
        const float *f = (const float *)mxGetData(prhs[1]);
        const float *x = (const float *)mxGetData(prhs[2]);

        residualf(r, f, x, G, h, sz);

    } else {
        double *r = (double *)mxGetData(prhs[0]);
        const double *f = (const double *)mxGetData(prhs[1]);
        const double *x = (const double *)mxGetData(prhs[2]);

        residuald(r, f, x, G, h, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}
#endif


void
mx_residual(mxArray *mxr,
            const mxArray *mxf, const mxArray *mxx, const mxArray *mxG,
            const mxArray *mxh)
{
    const uint8_t *G = (const uint8_t *)mxGetData(mxG);
    const double *h = (const double *)mxGetData(mxh);

    const size_t *sz = (const size_t *)mxGetDimensions(mxf);

    if (mxIsSingle(mxr)) {
        float *r = (float *)mxGetData(mxr);
        const float *f = (const float *)mxGetData(mxf);
        const float *x = (const float *)mxGetData(mxx);

        residualf(r, f, x, G, h, sz);

    } else {
        double *r = (double *)mxGetData(mxr);
        const double *f = (const double *)mxGetData(mxf);
        const double *x = (const double *)mxGetData(mxx);

        residuald(r, f, x, G, h, sz);
    }

    return;
}


void
residualf(float *r,
          const float *f, const float *x, const uint8_t *G,
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

    const float hx = (float)(1.0/(h[0]*h[0]));
    const float hy = (float)(1.0/(h[1]*h[1]));
    const float hz = (float)(1.0/(h[2]*h[2]));
    const float hh = (float)(-2.0*(hx+hy+hz));

    #pragma omp parallel for private(i,j,k,l) schedule(static) \
        if (nxny*nz > 32*32*32)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                if (G[l]) {
                    r[l] = f[l] +
                        (hh*x[l] +
                        hx*(x[l-1] + x[l+1]) +
                        hy*(x[l-nx] + x[l+nx]) +
                        hz*(x[l-nxny] + x[l+nxny]));
                }
            }
        }
    }

    return;
}


void
residuald(double *r,
          const double *f, const double *x, const uint8_t *G,
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

    const double hx = 1.0/(h[0]*h[0]);
    const double hy = 1.0/(h[1]*h[1]);
    const double hz = 1.0/(h[2]*h[2]);
    const double hh = -2.0*(hx+hy+hz);

    #pragma omp parallel for private(i,j,k,l) schedule(static) \
        if (nxny*nz > 32*32*32)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                if (G[l]) {
                    r[l] = f[l] +
                        (hh*x[l] +
                        hx*(x[l-1] + x[l+1]) +
                        hy*(x[l-nx] + x[l+nx]) +
                        hz*(x[l-nxny] + x[l+nxny]));
                }
            }
        }
    }

    return;
}
