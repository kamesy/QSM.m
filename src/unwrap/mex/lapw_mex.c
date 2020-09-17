#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include "mex.h"


#define PI 3.14159265358979323846
#define TAU (2*PI)

#define WRAPF(u) (((u) < -(float)PI) || ((u) > (float)PI))
#define WRAPD(u) (((u) < -(double)PI) || ((u) > (double)PI))


void lapwf(float *du, const float *u, const double *h, const size_t *sz);
void lapwd(double *du, const double *u, const double *h, const size_t *sz);

float wraptopif(float x);
double wraptopid(double x);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 3) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: lapw_mex(d2u, u, h);");
        return;
    }

    const double *h = (const double *)mxGetData(prhs[2]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *du = (float *)mxGetData(prhs[0]);
        const float *u = (const float *)mxGetData(prhs[1]);

        lapwf(du, u, h, sz);

    } else {
        double *du = (double *)mxGetData(prhs[0]);
        const double *u = (const double *)mxGetData(prhs[1]);

        lapwd(du, u, h, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}


void
lapwf(float *du, const float *u, const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    float ub, uf;

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

    #pragma omp parallel for private(i,j,k,l,ub,uf) schedule(static) \
        if(nxny*nz > 16*16*16)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                ub = u[l] - u[l-1];
                uf = u[l+1] - u[l];

                ub = WRAPF(ub) ? wraptopif(ub) : ub;
                uf = WRAPF(uf) ? wraptopif(uf) : uf;

                du[l] = hx*(uf - ub);

                ub = u[l] - u[l-nx];
                uf = u[l+nx] - u[l];

                ub = WRAPF(ub) ? wraptopif(ub) : ub;
                uf = WRAPF(uf) ? wraptopif(uf) : uf;

                du[l] = du[l] + hy*(uf - ub);

                ub = u[l] - u[l-nxny];
                uf = u[l+nxny] - u[l];

                ub = WRAPF(ub) ? wraptopif(ub) : ub;
                uf = WRAPF(uf) ? wraptopif(uf) : uf;

                du[l] = du[l] + hz*(uf - ub);
            }
        }
    }

    return;
}


void
lapwd(double *du, const double *u, const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    double ub, uf;

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

    #pragma omp parallel for private(i,j,k,l,ub,uf) schedule(static) \
        if(nxny*nz > 16*16*16)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                ub = u[l] - u[l-1];
                uf = u[l+1] - u[l];

                ub = WRAPD(ub) ? wraptopid(ub) : ub;
                uf = WRAPD(uf) ? wraptopid(uf) : uf;

                du[l] = hx*(uf - ub);

                ub = u[l] - u[l-nx];
                uf = u[l+nx] - u[l];

                ub = WRAPD(ub) ? wraptopid(ub) : ub;
                uf = WRAPD(uf) ? wraptopid(uf) : uf;

                du[l] = du[l] + hy*(uf - ub);

                ub = u[l] - u[l-nxny];
                uf = u[l+nxny] - u[l];

                ub = WRAPD(ub) ? wraptopid(ub) : ub;
                uf = WRAPD(uf) ? wraptopid(uf) : uf;

                du[l] = du[l] + hz*(uf - ub);
            }
        }
    }

    return;
}


inline float
wraptopif(float x)
{
    x += (float)PI;
    return x < 0.0f
        ? fmodf(x, (float)TAU) + (float)PI
        : fmodf(x, (float)TAU) - (float)PI;
}


inline double
wraptopid(double x)
{
    x += (double)PI;
    return x < 0.0
        ? fmod(x, (double)TAU) + (double)PI
        : fmod(x, (double)TAU) - (double)PI;
}
