#include <inttypes.h>
#include <omp.h>
#include "mex.h"


/* used to disable compiler warnings */
#define UNUSED(x) (void)(x)


void lapf(float *du,
          const float *u, const uint8_t *G,
          const double *h, const size_t *sz);

void lapd(double *du,
          const double *u, const uint8_t *G,
          const double *h, const size_t *sz);

float cf(const float *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G);
float ff(const float *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G);
float bf(const float *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G);
float of(const float *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G);

double cd(const double *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G);
double fd(const double *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G);
double bd(const double *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G);
double od(const double *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 4) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: lap_mex(d2u, u, G, h);");
        return;
    }

    const uint8_t *G = (const uint8_t *)mxGetData(prhs[2]);
    const double *h = (const double *)mxGetData(prhs[3]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *du = (float *)mxGetData(prhs[0]);
        const float *u = (const float *)mxGetData(prhs[1]);

        lapf(du, u, G, h, sz);

    } else {
        double *du = (double *)mxGetData(prhs[0]);
        const double *u = (const double *)mxGetData(prhs[1]);

        lapd(du, u, G, h, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}


void
lapf(float *du,
     const float *u, const uint8_t *G,
     const double *h, const size_t *sz)
{
    size_t i, j, k, l;

    uint8_t idx;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];

    const size_t nxny = nx*ny;
    const size_t nxnynz = nxny*nz;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const size_t NXX = nx-2;
    const size_t NYY = nx*(ny-2);
    const size_t NZZ = nxny*(nz-2);

    const float hx = (float)(1.0/(h[0]*h[0]));
    const float hy = (float)(1.0/(h[1]*h[1]));
    const float hz = (float)(1.0/(h[2]*h[2]));

    /*
     *  2*G[i+a] + G[i-a]:
     *      0 -> (o)
     *      1 -> (b)ackward
     *      2 -> (f)orward
     *      3 -> (c)entral
     */
    float(* const func_pt[])(const float *u,
                             size_t l, size_t a, size_t N, size_t i,
                             const uint8_t *G) = {of, bf, ff, cf};


    #pragma omp parallel for private(i,j,k,l,idx) schedule(static) \
        if(nxny*nz > 16*16*16)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = j + k;
            for(i = 0; i < nx; ++i, ++l) {
                if (G[l]) {
                    idx = (i-1) < NXX
                        ? 2*G[l+1] + G[l-1]
                        : (i == 0)*2 + (i == NX);

                    du[l] = hx*(*func_pt[idx])(u, l, 1, nx, i, G);

                    idx = (j-nx) < NYY
                        ? 2*G[l+nx] + G[l-nx]
                        : (j == 0)*2 + (j == NY);

                    du[l] += hy*(*func_pt[idx])(u, l, nx, nxny, j, G);

                    idx = (k-nxny) < NZZ
                        ? 2*G[l+nxny] + G[l-nxny]
                        : (k == 0)*2 + (k == NZ);

                    du[l] += hz*(*func_pt[idx])(u, l, nxny, nxnynz, k, G);
                }
            }
        }
    }

    return;
}


void
lapd(double *du,
     const double *u, const uint8_t *G,
     const double *h, const size_t *sz)
{
    size_t i, j, k, l;
    uint8_t idx;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];

    const size_t nxny = nx*ny;
    const size_t nxnynz = nxny*nz;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const size_t NXX = nx-2;
    const size_t NYY = nx*(ny-2);
    const size_t NZZ = nxny*(nz-2);

    const double hx = 1.0/(h[0]*h[0]);
    const double hy = 1.0/(h[1]*h[1]);
    const double hz = 1.0/(h[2]*h[2]);

    /*
     *  2*G[i+a] + G[i-a]:
     *      0 -> (o)
     *      1 -> (b)ackward
     *      2 -> (f)orward
     *      3 -> (c)entral
     */
    double(* const func_pt[])(const double *u,
                              size_t l, size_t a, size_t N, size_t i,
                              const uint8_t *G) = {od, bd, fd, cd};


    #pragma omp parallel for private(i,j,k,l,idx) schedule(static) \
        if(nxny*nz > 16*16*16)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = j + k;
            for(i = 0; i < nx; ++i, ++l) {
                if (G[l]) {
                    idx = (i-1) < NXX
                        ? 2*G[l+1] + G[l-1]
                        : (i == 0)*2 + (i == NX);

                    du[l] = hx*(*func_pt[idx])(u, l, 1, nx, i, G);

                    idx = (j-nx) < NYY
                        ? 2*G[l+nx] + G[l-nx]
                        : (j == 0)*2 + (j == NY);

                    du[l] += hy*(*func_pt[idx])(u, l, nx, nxny, j, G);

                    idx = (k-nxny) < NZZ
                        ? 2*G[l+nxny] + G[l-nxny]
                        : (k == 0)*2 + (k == NZ);

                    du[l] += hz*(*func_pt[idx])(u, l, nxny, nxnynz, k, G);
                }
            }
        }
    }

    return;
}


float
cf(const float *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G)
{
    /* disable compiler warnings */
    UNUSED(N);
    UNUSED(i);
    UNUSED(G);
    return u[l-a] - 2.0f*u[l] + u[l+a];
}


float
ff(const float *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G)
{
    if ((i+3*a < N) && G[l+2*a] && G[l+3*a]) {
        return 2.0f*u[l] - 5.0f*u[l+a] + 4.0f*u[l+2*a] - u[l+3*a];

    } else if ((i+2*a < N) && G[l+2*a]) {
        return u[l] - 2.0f*u[l+a] + u[l+2*a];

    } else {
        return -u[l] + u[l+a];
    }
}


float
bf(const float *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G)
{
    if ((i-3*a < N) && G[l-3*a] && G[l-2*a]) {
        return -u[l-3*a] + 4.0f*u[l-2*a] - 5.0f*u[l-a] + 2.0f*u[l];

    } else if ((i-2*a < N) && G[l-2*a]) {
        return u[l-2*a] - 2.0f*u[l-a] + u[l];

    } else {
        return u[l-a] - u[l];
    }
}


float
of(const float *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G)
{
    /* disable compiler warnings */
    UNUSED(u);
    UNUSED(l);
    UNUSED(a);
    UNUSED(N);
    UNUSED(i);
    UNUSED(G);
    return 0.0f;
}


double
cd(const double *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G)
{
    /* disable compiler warnings */
    UNUSED(N);
    UNUSED(i);
    UNUSED(G);
    return u[l-a] -2.0*u[l] + u[l+a];
}


double
fd(const double *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G)
{
    if ((i+3*a < N) && G[l+2*a] && G[l+3*a]) {
        return 2.0*u[l] - 5.0*u[l+a] + 4.0*u[l+2*a] - u[l+3*a];

    } else if ((i+2*a < N) && G[l+2*a]) {
        return u[l] - 2.0*u[l+a] + u[l+2*a];

    } else {
        return u[l+a] - u[l];
    }
}


double
bd(const double *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G)
{
    if ((i-3*a < N) && G[l-3*a] && G[l-2*a]) {
        return -u[l-3*a] + 4.0*u[l-2*a] - 5.0*u[l-a] + 2.0*u[l];

    } else if ((i-2*a < N) && G[l-2*a]) {
        return u[l-2*a] - 2.0*u[l-a] + u[l];

    } else {
        return u[l-a] - u[l];
    }
}


double
od(const double *u, size_t l, size_t a, size_t N, size_t i, const uint8_t *G)
{
    /* disable compiler warnings */
    UNUSED(u);
    UNUSED(l);
    UNUSED(a);
    UNUSED(N);
    UNUSED(i);
    UNUSED(G);
    return 0.0;
}
