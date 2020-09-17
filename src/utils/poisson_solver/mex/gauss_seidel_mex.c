#include <inttypes.h>
#include <omp.h>
#include "mex.h"
#include "gauss_seidel_mex.h"


void gauss_seidelf(float *v,
                   const float *f, const uint8_t *G,
                   const double *h, const size_t *sz,
                   int32_t iter, const uint8_t rev);

void gauss_seideld(double *v,
                   const double *f, const uint8_t *G,
                   const double *h, const size_t *sz,
                   int32_t iter, const uint8_t rev);


#ifdef GAUSS_SEIDEL_MEX
void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 6) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: gauss_seidel_mex(v, f, G, h, iter, reverse);");
    }

    const uint8_t *G = (const uint8_t *)mxGetData(prhs[2]);
    const double *h = (const double *)mxGetData(prhs[3]);
    int32_t iter = (int32_t)mxGetScalar(prhs[4]);
    uint8_t reverse = (uint8_t)mxGetScalar(prhs[5]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);

    if (mxIsSingle(prhs[0])) {
        float *v = (float *)mxGetData(prhs[0]);
        const float *f = (const float *)mxGetData(prhs[1]);
        gauss_seidelf(v, f, G, h, sz, iter, reverse);

    } else {
        double *v = (double *)mxGetData(prhs[0]);
        const double *f = (const double *)mxGetData(prhs[1]);
        gauss_seideld(v, f, G, h, sz, iter, reverse);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}
#endif


void
mx_gauss_seidel(mxArray *mxv,
                const mxArray *mxf, const mxArray *mxG,
                const mxArray *mxh, int32_t iter, const uint8_t rev)
{
    const uint8_t *G = (const uint8_t *)mxGetData(mxG);
    const double *h = (const double *)mxGetData(mxh);

    const size_t *sz = (const size_t *)mxGetDimensions(mxf);

    if (mxIsSingle(mxv)) {
        float *v = (float *)mxGetData(mxv);
        const float *f = (const float *)mxGetData(mxf);
        gauss_seidelf(v, f, G, h, sz, iter, rev);

    } else {
        double *v = (double *)mxGetData(mxv);
        const double *f = (const double *)mxGetData(mxf);
        gauss_seideld(v, f, G, h, sz, iter, rev);
    }

    return;
}


void
gauss_seidelf(float *v,
              const float *f, const uint8_t *G,
              const double *h, const size_t *sz,
              int32_t iter, const uint8_t rev)
{
    int32_t i, j, k;
    int32_t l;

    const int32_t nx = (int32_t)sz[0];
    const int32_t ny = (int32_t)sz[1];
    const int32_t nz = (int32_t)sz[2];
    const int32_t nxny = nx*ny;

    const int32_t NX = rev ? nx-2 : nx-1;
    const int32_t NY = rev ? nx*(ny-2) : nx*(ny-1);
    const int32_t NZ = rev ? nxny*(nz-2) : nxny*(nz-1);

    const float hx = (float)(1.0/(h[0]*h[0]));
    const float hy = (float)(1.0/(h[1]*h[1]));
    const float hz = (float)(1.0/(h[2]*h[2]));
    const float hh = (float)(1.0/(2.0*(double)(hx+hy+hz)));

#ifndef GAUSS_SEIDEL_RED_BLACK
    if (rev > 0) {
        while(iter-- > 0) {
            for(k = NZ; k >= nxny; k -= nxny) {
                for(j = NY; j >= nx; j -= nx) {
                    l = NX + j + k;
                    for(i = NX; i >= 1; --i, --l) {
                        if (G[l]) {
                            v[l] = hh *
                                (f[l] +
                                (hx*(v[l-1] + v[l+1]) +
                                hy*(v[l-nx] + v[l+nx]) +
                                hz*(v[l-nxny] + v[l+nxny])));
                        }
                    }
                }
            }
        }
    } else {
        while(iter-- > 0) {
            for(k = nxny; k < NZ; k += nxny) {
                for(j = nx; j < NY; j += nx) {
                    l = 1 + j + k;
                    for(i = 1; i < NX; ++i, ++l) {
                        if (G[l]) {
                            v[l] = hh *
                                (f[l] +
                                (hx*(v[l-1] + v[l+1]) +
                                hy*(v[l-nx] + v[l+nx]) +
                                hz*(v[l-nxny] + v[l+nxny])));
                        }
                    }
                }
            }
        }
    }

#else
    size_t s, is, js, ks;

    if (rev > 0) {
        while(iter-- > 0) {
            for(s = 0; s < 2; ++s) {
                #pragma omp parallel for private(i,j,k,l,is,js,ks) \
                    schedule(static) if (nxny*nz > 32*32*32)
                for(k = NZ; k >= nxny; k -= nxny) {
                    ks = (k/nxny) & 1;
                    for(j = NY; j >= nx; j -= nx) {
                        js = (j/nx) & 1;
                        is = s + ((ks && js) || !(js || ks));
                        l = NX-is + j + k;
                        for(i = NX-is; i >= 1; i -= 2, l -= 2) {
                            if (G[l]) {
                                v[l] = hh *
                                    (f[l] +
                                    (hx*(v[l-1] + v[l+1]) +
                                    hy*(v[l-nx] + v[l+nx]) +
                                    hz*(v[l-nxny] + v[l+nxny])));
                            }
                        }
                    }
                }
            }
        }
    } else {
        while(iter-- > 0) {
            for(s = 0; s < 2; ++s) {
                #pragma omp parallel for private(i,j,k,l,is,js,ks) \
                    schedule(static) if (nxny*nz > 32*32*32)
                for(k = nxny; k < NZ; k += nxny) {
                    ks = (k/nxny) & 1;
                    for(j = nx; j < NY; j += nx) {
                        js = (j/nx) & 1;
                        is = s + ((ks && js) || !(js || ks));
                        l = 1 + is + j + k;
                        for(i = 1 + is; i < NX; i += 2, l += 2) {
                            if (G[l]) {
                                v[l] = hh *
                                    (f[l] +
                                    (hx*(v[l-1] + v[l+1]) +
                                    hy*(v[l-nx] + v[l+nx]) +
                                    hz*(v[l-nxny] + v[l+nxny])));
                            }
                        }
                    }
                }
            }
        }
    }
#endif

    return;
}


void
gauss_seideld(double *v,
              const double *f, const uint8_t *G,
              const double *h, const size_t *sz,
              int32_t iter, const uint8_t rev)
{
    int32_t i, j, k;
    int32_t l;

    const int32_t nx = (int32_t)sz[0];
    const int32_t ny = (int32_t)sz[1];
    const int32_t nz = (int32_t)sz[2];
    const int32_t nxny = nx*ny;

    const int32_t NX = rev ? nx-2 : nx-1;
    const int32_t NY = rev ? nx*(ny-2) : nx*(ny-1);
    const int32_t NZ = rev ? nxny*(nz-2) : nxny*(nz-1);

    const double hx = 1.0/(h[0]*h[0]);
    const double hy = 1.0/(h[1]*h[1]);
    const double hz = 1.0/(h[2]*h[2]);
    const double hh = 1.0/(2.0*(hx+hy+hz));

#ifndef GAUSS_SEIDEL_RED_BLACK
    if (rev > 0) {
        while(iter-- > 0) {
            for(k = NZ; k >= nxny; k -= nxny) {
                for(j = NY; j >= nx; j -= nx) {
                    l = NX + j + k;
                    for(i = NX; i >= 1; --i, --l) {
                        if (G[l]) {
                            v[l] = hh *
                                (f[l] +
                                (hx*(v[l-1] + v[l+1]) +
                                hy*(v[l-nx] + v[l+nx]) +
                                hz*(v[l-nxny] + v[l+nxny])));
                        }
                    }
                }
            }
        }
    } else {
        while(iter-- > 0) {
            for(k = nxny; k < NZ; k += nxny) {
                for(j = nx; j < NY; j += nx) {
                    l = 1 + j + k;
                    for(i = 1; i < NX; ++i, ++l) {
                        if (G[l]) {
                            v[l] = hh *
                                (f[l] +
                                (hx*(v[l-1] + v[l+1]) +
                                hy*(v[l-nx] + v[l+nx]) +
                                hz*(v[l-nxny] + v[l+nxny])));
                        }
                    }
                }
            }
        }
    }

#else
    size_t s, is, js, ks;

    if (rev > 0) {
        while(iter-- > 0) {
            for(s = 0; s < 2; ++s) {
                #pragma omp parallel for private(i,j,k,l,is,js,ks) \
                    schedule(static) if (nxny*nz > 32*32*32)
                for(k = NZ; k >= nxny; k -= nxny) {
                    ks = (k/nxny) & 1;
                    for(j = NY; j >= nx; j -= nx) {
                        js = (j/nx) & 1;
                        is = s + ((ks && js) || !(js || ks));
                        l = NX-is + j + k;
                        for(i = NX-is; i >= 1; i -= 2, l -= 2) {
                            if (G[l]) {
                                v[l] = hh *
                                    (f[l] +
                                    (hx*(v[l-1] + v[l+1]) +
                                    hy*(v[l-nx] + v[l+nx]) +
                                    hz*(v[l-nxny] + v[l+nxny])));
                            }
                        }
                    }
                }
            }
        }
    } else {
        while(iter-- > 0) {
            for(s = 0; s < 2; ++s) {
                #pragma omp parallel for private(i,j,k,l,is,js,ks) \
                    schedule(static) if (nxny*nz > 32*32*32)
                for(k = nxny; k < NZ; k += nxny) {
                    ks = (k/nxny) & 1;
                    for(j = nx; j < NY; j += nx) {
                        js = (j/nx) & 1;
                        is = s + ((ks && js) || !(js || ks));
                        l = 1 + is + j + k;
                        for(i = 1 + is; i < NX; i += 2, l += 2) {
                            if (G[l]) {
                                v[l] = hh *
                                    (f[l] +
                                    (hx*(v[l-1] + v[l+1]) +
                                    hy*(v[l-nx] + v[l+nx]) +
                                    hz*(v[l-nxny] + v[l+nxny])));
                            }
                        }
                    }
                }
            }
        }
    }
#endif

    return;
}
