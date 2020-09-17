#include <inttypes.h>
#include <math.h>
#if 0
#include <omp.h> /* TODO */
#endif
#include "mex.h"
#include "norm2_mex.h"


#define BLOCK_SIZE 128


float norm2f(const float *x, const uint8_t *G, const size_t n);
double norm2d(const double *x, const uint8_t *G, const size_t n);

float norm22f(const float *x, const uint8_t *G, const size_t n);
double norm22d(const double *x, const uint8_t *G, const size_t n);


#ifdef NORM2_MEX
void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 2) || (nlhs > 1)) {
        mexErrMsgTxt("Usage: p = norm2_mex(x, G);");
        return;
    }

    const uint8_t *G = (const uint8_t *)mxGetData(prhs[1]);
    const size_t n = mxGetNumberOfElements(prhs[0]);

    const size_t one[1] = {1};

    if (mxIsSingle(prhs[0])) {
        plhs[0] = mxCreateNumericArray(1, one, mxSINGLE_CLASS, mxREAL);
        float *x = (float *)mxGetData(prhs[0]);

        *(float *)mxGetData(plhs[0]) = norm2f(x, G, n);

    } else if (mxIsDouble(prhs[0])) {
        plhs[0] = mxCreateNumericArray(1, one, mxDOUBLE_CLASS, mxREAL);
        double *x = (double *)mxGetData(prhs[0]);

        *(double *)mxGetData(plhs[0]) = norm2d(x, G, n);

    } else {
        plhs[0] = mxCreateDoubleScalar(-1.0);
    }

    return;
}
#endif


double
mx_norm2(const mxArray *mxx, const mxArray *mxG)
{
    double p = -1.0;

    const uint8_t *G = (const uint8_t *)mxGetData(mxG);
    const size_t n = mxGetNumberOfElements(mxx);

    if (mxIsSingle(mxx)) {
        const float *x = (const float *)mxGetData(mxx);
        p = (double)norm2f(x, G, n);

    } else {
        const double *x = (const double *)mxGetData(mxx);
        p = norm2d(x, G, n);
    }

    return p;
}


float
norm2f(const float *x, const uint8_t *G, const size_t n)
{
    return sqrtf(norm22f(x, G, n));
}


float
norm22f(const float *x, const uint8_t *G, const size_t n)
{
    float s;

    if (n <= BLOCK_SIZE) {
        s = G[0] ? x[0]*x[0] : 0.0f;
        for(size_t i = 1; i < n; ++i) {
            if (G[i]) {
                s += x[i]*x[i];
            }
        }

    } else {
        size_t m = n >> 1;
        s = norm22f(x, G, m) + norm22f(x+m, G+m, n-m);
    }

    return s;
}


double
norm2d(const double *x, const uint8_t *G, const size_t n)
{
    return sqrt(norm22d(x, G, n));
}


double
norm22d(const double *x, const uint8_t *G, const size_t n)
{
    double s;

    if (n <= BLOCK_SIZE) {
        s = G[0] ? x[0]*x[0] : 0.0;
        for(size_t i = 1; i < n; ++i) {
            if (G[i]) {
                s += x[i]*x[i];
            }
        }

    } else {
        size_t m = n >> 1;
        s = norm22d(x, G, m) + norm22d(x+m, G+m, n-m);
    }

    return s;
}
