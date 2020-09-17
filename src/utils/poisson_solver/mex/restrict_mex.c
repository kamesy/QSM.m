#include <inttypes.h>
#include <omp.h>
#include "mex.h"
#include "restrict_mex.h"


void restrictf(float *x2,
               const float *x, const uint8_t *G2,
               const size_t *sz2, const size_t *sz);

void restrictd(double *x2,
               const double *x, const uint8_t *G2,
               const size_t *sz2, const size_t *sz);


#ifdef RESTRICT_MEX
void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 3) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: restrict_mex(x2, x, G2);");
    }

    const uint8_t *G2 = (const uint8_t *)mxGetData(prhs[2]);

    const size_t *sz2 = (const size_t *)mxGetDimensions(prhs[0]);
    const size_t *sz = (const size_t *)mxGetDimensions(prhs[1]);

    if (mxIsSingle(prhs[0])) {
        float *x2 = (float *)mxGetData(prhs[0]);
        const float *x = (const float *)mxGetData(prhs[1]);
        restrictf(x2, x, G2, sz2, sz);

    } else {
        double *x2 = (double *)mxGetData(prhs[0]);
        const double *x = (const double *)mxGetData(prhs[1]);
        restrictd(x2, x, G2, sz2, sz);
    }

    if (nlhs == 1) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    return;
}
#endif


void
mx_restrict(mxArray *mxx2, const mxArray *mxx, const mxArray *mxG2)
{
    const uint8_t *G2 = (const uint8_t *)mxGetData(mxG2);

    const size_t *sz2 = (const size_t *)mxGetDimensions(mxx2);
    const size_t *sz = (const size_t *)mxGetDimensions(mxx);

    if (mxIsSingle(mxx2)) {
        float *x2 = (float *)mxGetData(mxx2);
        const float *x = (const float *)mxGetData(mxx);
        restrictf(x2, x, G2, sz2, sz);

    } else {
        double *x2 = (double *)mxGetData(mxx2);
        const double *x = (const double *)mxGetData(mxx);
        restrictd(x2, x, G2, sz2, sz);
    }

    return;
}


void
restrictf(float *x2,
          const float *x, const uint8_t *G2,
          const size_t *sz2, const size_t *sz)
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

    /*
      1/64 *
        [:, :, 1] =
         1  2  1
         2  4  2
         1  2  1

        [:, :, 2] =
         2  4  2
         4  8  4
         2  4  2

        [:, :, 3] =
         1  2  1
         2  4  2
         1  2  1
    */
    const size_t o111 = -1  -nx  -nxny; /* 1 */
    const size_t o211 =  0  -nx  -nxny; /* 2 */
    const size_t o311 =  1  -nx  -nxny; /* 1 */

    const size_t o121 = -1   +0  -nxny; /* 2 */
    const size_t o221 =  0   +0  -nxny; /* 4 */
    const size_t o321 =  1   +0  -nxny; /* 2 */

    const size_t o131 = -1  +nx  -nxny; /* 1 */
    const size_t o231 =  0  +nx  -nxny; /* 2 */
    const size_t o331 =  1  +nx  -nxny; /* 1 */

    const size_t o112 = -1  -nx     +0; /* 2 */
    const size_t o212 =  0  -nx     +0; /* 4 */
    const size_t o312 =  1  -nx     +0; /* 2 */

    const size_t o122 = -1   +0     +0; /* 4 */
    const size_t o322 =  1   +0     +0; /* 4 */

    const size_t o132 = -1  +nx     +0; /* 2 */
    const size_t o232 =  0  +nx     +0; /* 4 */
    const size_t o332 =  1  +nx     +0; /* 2 */

    const size_t o113 = -1  -nx  +nxny; /* 1 */
    const size_t o213 =  0  -nx  +nxny; /* 2 */
    const size_t o313 =  1  -nx  +nxny; /* 1 */

    const size_t o123 = -1   +0  +nxny; /* 2 */
    const size_t o223 =  0   +0  +nxny; /* 4 */
    const size_t o323 =  1   +0  +nxny; /* 2 */

    const size_t o133 = -1  +nx  +nxny; /* 1 */
    const size_t o233 =  0  +nx  +nxny; /* 2 */
    const size_t o333 =  1  +nx  +nxny; /* 1 */


    #pragma omp parallel for private(i2,j2,k2,l2,lk2,l,lk) schedule(static) \
        if (nxny*nz > 32*32*32)
    for(k2 = 1; k2 < NZ2; ++k2) {
        lk2 = nxny2*k2;
        lk = nxny*((k2<<1)-1);

        for(j2 = 1; j2 < NY2; ++j2) {
            l2 = 1 + nx2*j2 + lk2;
            l = 1 + nx*((j2<<1)-1) + lk;

            for(i2 = 1; i2 < NX2; ++i2, ++l2, l += 2) {

                x2[l2] = G2[l2] ?
                    0.015625f * (
                        x[l+o111] +
                        x[l+o311] +
                        x[l+o131] +
                        x[l+o331] +
                        x[l+o113] +
                        x[l+o313] +
                        x[l+o133] +
                        x[l+o333]
                    ) +
                    0.03125f * (
                        x[l+o211] +
                        x[l+o121] +
                        x[l+o321] +
                        x[l+o231] +
                        x[l+o312] +
                        x[l+o112] +
                        x[l+o132] +
                        x[l+o332] +
                        x[l+o213] +
                        x[l+o123] +
                        x[l+o323] +
                        x[l+o233]
                    ) +
                    0.0625f * (
                        x[l+o221] +
                        x[l+o212] +
                        x[l+o122] +
                        x[l+o322] +
                        x[l+o232] +
                        x[l+o223]
                    ) +
                    0.125f * x[l]
                    : 0.0f;
            }
        }
    }

    return;
}


void
restrictd(double *x2,
          const double *x, const uint8_t *G2,
          const size_t *sz2, const size_t *sz)
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

    /*
      1/64 *
        [:, :, 1] =
         1  2  1
         2  4  2
         1  2  1

        [:, :, 2] =
         2  4  2
         4  8  4
         2  4  2

        [:, :, 3] =
         1  2  1
         2  4  2
         1  2  1
    */
    const size_t o111 = -1  -nx  -nxny; /* 1 */
    const size_t o211 =  0  -nx  -nxny; /* 2 */
    const size_t o311 =  1  -nx  -nxny; /* 1 */

    const size_t o121 = -1   +0  -nxny; /* 2 */
    const size_t o221 =  0   +0  -nxny; /* 4 */
    const size_t o321 =  1   +0  -nxny; /* 2 */

    const size_t o131 = -1  +nx  -nxny; /* 1 */
    const size_t o231 =  0  +nx  -nxny; /* 2 */
    const size_t o331 =  1  +nx  -nxny; /* 1 */

    const size_t o112 = -1  -nx     +0; /* 2 */
    const size_t o212 =  0  -nx     +0; /* 4 */
    const size_t o312 =  1  -nx     +0; /* 2 */

    const size_t o122 = -1   +0     +0; /* 4 */
    const size_t o322 =  1   +0     +0; /* 4 */

    const size_t o132 = -1  +nx     +0; /* 2 */
    const size_t o232 =  0  +nx     +0; /* 4 */
    const size_t o332 =  1  +nx     +0; /* 2 */

    const size_t o113 = -1  -nx  +nxny; /* 1 */
    const size_t o213 =  0  -nx  +nxny; /* 2 */
    const size_t o313 =  1  -nx  +nxny; /* 1 */

    const size_t o123 = -1   +0  +nxny; /* 2 */
    const size_t o223 =  0   +0  +nxny; /* 4 */
    const size_t o323 =  1   +0  +nxny; /* 2 */

    const size_t o133 = -1  +nx  +nxny; /* 1 */
    const size_t o233 =  0  +nx  +nxny; /* 2 */
    const size_t o333 =  1  +nx  +nxny; /* 1 */


    #pragma omp parallel for private(i2,j2,k2,l2,lk2,l,lk) schedule(static) \
        if (nxny*nz > 32*32*32)
    for(k2 = 1; k2 < NZ2; ++k2) {
        lk2 = nxny2*k2;
        lk = nxny*((k2<<1)-1);

        for(j2 = 1; j2 < NY2; ++j2) {
            l2 = 1 + nx2*j2 + lk2;
            l = 1 + nx*((j2<<1)-1) + lk;

            for(i2 = 1; i2 < NX2; ++i2, ++l2, l += 2) {

                x2[l2] = G2[l2] ?
                    0.015625 * (
                        x[l+o111] +
                        x[l+o311] +
                        x[l+o131] +
                        x[l+o331] +
                        x[l+o113] +
                        x[l+o313] +
                        x[l+o133] +
                        x[l+o333]
                    ) +
                    0.03125 * (
                        x[l+o211] +
                        x[l+o121] +
                        x[l+o321] +
                        x[l+o231] +
                        x[l+o312] +
                        x[l+o112] +
                        x[l+o132] +
                        x[l+o332] +
                        x[l+o213] +
                        x[l+o123] +
                        x[l+o323] +
                        x[l+o233]
                    ) +
                    0.0625 * (
                        x[l+o221] +
                        x[l+o212] +
                        x[l+o122] +
                        x[l+o322] +
                        x[l+o232] +
                        x[l+o223]
                    ) +
                    0.125 * x[l]
                    : 0.0;
            }
        }
    }

    return;
}
