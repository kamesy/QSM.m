#include <inttypes.h>
#include <omp.h>
#include "mex.h"
#include "prolong_mex.h"


void prolongf(float *x,
              const float *x2, const uint8_t *G,
              const size_t *sz, const size_t *sz2);

void prolongd(double *x,
              const double *x2, const uint8_t *G,
              const size_t *sz, const size_t *sz2);


#ifdef PROLONG_MEX
void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 3) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: prolong_mex(x, x2, G);");
    }

    const uint8_t *G = (const uint8_t *)mxGetData(prhs[2]);

    const size_t *sz = (const size_t *)mxGetDimensions(prhs[0]);
    const size_t *sz2 = (const size_t *)mxGetDimensions(prhs[1]);

    if (mxIsSingle(prhs[0])) {
        float *x = (float *)mxGetData(prhs[0]);
        const float *x2 = (const float *)mxGetData(prhs[1]);
        prolongf(x, x2, G, sz, sz2);

    } else {
        double *x = (double *)mxGetData(prhs[0]);
        const double *x2 = (const double *)mxGetData(prhs[1]);
        prolongd(x, x2, G, sz, sz2);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}
#endif


void
mx_prolong(mxArray *mxx, const mxArray *mxx2, const mxArray *mxG)
{
    const uint8_t *G = (const uint8_t *)mxGetData(mxG);

    const size_t *sz = (const size_t *)mxGetDimensions(mxx);
    const size_t *sz2 = (const size_t *)mxGetDimensions(mxx2);

    if (mxIsSingle(mxx)) {
        float *x = (float *)mxGetData(mxx);
        const float *x2 = (const float *)mxGetData(mxx2);
        prolongf(x, x2, G, sz, sz2);

    } else {
        double *x = (double *)mxGetData(mxx);
        const double *x2 = (const double *)mxGetData(mxx2);
        const uint8_t *G = (const uint8_t *)mxGetData(mxG);
        prolongd(x, x2, G, sz, sz2);
    }

    return;
}


void
prolongf(float *x,
         const float *x2, const uint8_t *G,
         const size_t *sz, const size_t *sz2)
{
    size_t i2, j2, k2;
    size_t l2, lk2;
    size_t l, lk;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t nx2 = sz2[0];
    const size_t ny2 = sz2[1];
    const size_t nz2 = sz2[2];
    const size_t nxny2 = nx2*ny2;

    const size_t NX2 = nx2-1;
    const size_t NY2 = ny2-1;
    const size_t NZ2 = nz2-1;

    /* offset indices */
    const size_t o110 = 1  + nx  +    0;
    const size_t o101 = 1  +  0  + nxny;
    const size_t o011 = 0  + nx  + nxny;
    const size_t o111 = 1  + nx  + nxny;

    const size_t o110_2 = 1  + nx2  +     0;
    const size_t o101_2 = 1  +   0  + nxny2;
    const size_t o011_2 = 0  + nx2  + nxny2;
    const size_t o111_2 = 1  + nx2  + nxny2;


    #pragma omp parallel for private(i2,j2,k2,l2,lk2,l,lk) schedule(static) \
        if(nxny*nz > 32*32*32)
    for(k2 = 1; k2 < NZ2; ++k2) {
        lk2 = nxny2*k2;
        lk = nxny*((k2<<1)-1);

        for(j2 = 1; j2 < NY2; ++j2) {
            l2 = 1 + nx2*j2 + lk2;
            l = 1 + nx*((j2<<1)-1) + lk;

            for(i2 = 1; i2 < NX2; ++i2, ++l2, l += 2) {

                x[l] = G[l] ? x2[l2] : 0.0f;

                x[l+1] = G[l+1] ?
                    0.5f*(
                        x2[l2] +
                        x2[l2+1]
                    )
                    : 0.0f;

                x[l+nx] = G[l+nx] ?
                    0.5f*(
                        x2[l2] +
                        x2[l2+nx2]
                    )
                    : 0.0f;

                x[l+nxny] = G[l+nxny] ?
                    0.5f*(
                        x2[l2] +
                        x2[l2+nxny2]
                    )
                    : 0.0f;

                x[l+o110] = G[l+o110] ?
                    0.25f*(
                        x2[l2] +
                        x2[l2+1] +
                        x2[l2+nx2] +
                        x2[l2+o110_2]
                    )
                    : 0.0f;

                x[l+o101] = G[l+o101] ?
                    0.25f*(
                        x2[l2] +
                        x2[l2+1] +
                        x2[l2+nxny2] +
                        x2[l2+o101_2]
                    )
                    : 0.0f;

                x[l+o011] = G[l+o011] ?
                    0.25f*(
                        x2[l2] +
                        x2[l2+nx2] +
                        x2[l2+nxny2] +
                        x2[l2+o011_2]
                    )
                    : 0.0f;

                x[l+o111] = G[l+o111] ?
                    0.125f*(
                        x2[l2] +
                        x2[l2+1] +
                        x2[l2+nx2] +
                        x2[l2+nxny2] +
                        x2[l2+o110_2] +
                        x2[l2+o101_2] +
                        x2[l2+o011_2] +
                        x2[l2+o111_2]
                    )
                    : 0.0f;
            }
        }
    }

    return;
}


void
prolongd(double *x,
         const double *x2, const uint8_t *G,
         const size_t *sz, const size_t *sz2)
{
    size_t i2, j2, k2;
    size_t l2, lk2;
    size_t l, lk;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t nx2 = sz2[0];
    const size_t ny2 = sz2[1];
    const size_t nz2 = sz2[2];
    const size_t nxny2 = nx2*ny2;

    const size_t NX2 = nx2-1;
    const size_t NY2 = ny2-1;
    const size_t NZ2 = nz2-1;

    /* offset indices */
    const size_t o110 = 1  + nx  +    0;
    const size_t o101 = 1  +  0  + nxny;
    const size_t o011 = 0  + nx  + nxny;
    const size_t o111 = 1  + nx  + nxny;

    const size_t o110_2 = 1  + nx2  +     0;
    const size_t o101_2 = 1  +   0  + nxny2;
    const size_t o011_2 = 0  + nx2  + nxny2;
    const size_t o111_2 = 1  + nx2  + nxny2;


    #pragma omp parallel for private(i2,j2,k2,l2,lk2,l,lk) schedule(static) \
        if(nxny*nz > 32*32*32)
    for(k2 = 1; k2 < NZ2; ++k2) {
        lk2 = nxny2*k2;
        lk = nxny*((k2<<1)-1);

        for(j2 = 1; j2 < NY2; ++j2) {
            l2 = 1 + nx2*j2 + lk2;
            l = 1 + nx*((j2<<1)-1) + lk;

            for(i2 = 1; i2 < NX2; ++i2, ++l2, l += 2) {

                x[l] = G[l] ? x2[l2] : 0.0;

                x[l+1] = G[l+1] ?
                    0.5*(
                        x2[l2] +
                        x2[l2+1]
                    )
                    : 0.0;

                x[l+nx] = G[l+nx] ?
                    0.5*(
                        x2[l2] +
                        x2[l2+nx2]
                    )
                    : 0.0;

                x[l+nxny] = G[l+nxny] ?
                    0.5*(
                        x2[l2] +
                        x2[l2+nxny2]
                    )
                    : 0.0;

                x[l+o110] = G[l+o110] ?
                    0.25*(
                        x2[l2] +
                        x2[l2+1] +
                        x2[l2+nx2] +
                        x2[l2+o110_2]
                    )
                    : 0.0;

                x[l+o101] = G[l+o101] ?
                    0.25*(
                        x2[l2] +
                        x2[l2+1] +
                        x2[l2+nxny2] +
                        x2[l2+o101_2]
                    )
                    : 0.0;

                x[l+o011] = G[l+o011] ?
                    0.25*(
                        x2[l2] +
                        x2[l2+nx2] +
                        x2[l2+nxny2] +
                        x2[l2+o011_2]
                    )
                    : 0.0;

                x[l+o111] = G[l+o111] ?
                    0.125*(
                        x2[l2] +
                        x2[l2+1] +
                        x2[l2+nx2] +
                        x2[l2+nxny2] +
                        x2[l2+o110_2] +
                        x2[l2+o101_2] +
                        x2[l2+o011_2] +
                        x2[l2+o111_2]
                    )
                    : 0.0;
            }
        }
    }

    return;
}
