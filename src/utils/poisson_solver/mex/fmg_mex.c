#include <inttypes.h>
#include <string.h>
#include "mex.h"

#include "boundary_mask_mex.h"
#include "coarsen_grid_mex.h"
#include "correct_mex.h"
#include "gauss_seidel_mex.h"
#include "mx_blas.h"
#include "mx_util.h"
#include "prolong_mex.h"
#include "restrict_mex.h"
#include "residual_mex.h"


#define MAX_DEPTH 16


struct MG
{
    mxArray *f;
    mxArray *v;
    mxArray *r;
    mxArray *G;
    mxArray *B;
    mxArray *h;
};


struct Iter
{
    const int32_t mg;
    const int32_t mu;
    const int32_t pre;
    const int32_t post;
    const int32_t bs;
    const int32_t l;
    const double tol;
};


void mx_init(struct MG *mg,
             const mxArray *mxf,
             const mxArray *mxG,
             const mxArray *mxh,
             const int32_t l);

void mx_fmg(struct MG *mg, const struct Iter *iter, const int32_t l);
void mx_cycle(struct MG *mg, const struct Iter *iter, const int32_t l);

void mx_cleanup(struct MG *mg, const int32_t l);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 10) || (nlhs > 1)) {
        mexErrMsgTxt(
            "Usage: v = fmg_mex("
                "f, mask, vsz, "
                "tol, maxit, mu, npre, npost, nboundary, nlevels)"
        );
        return;
    }

    const struct Iter iter = {
        .mg = (int32_t)mxGetScalar(prhs[4]),
        .mu = (int32_t)mxGetScalar(prhs[5]),
        .pre = (int32_t)mxGetScalar(prhs[6]),
        .post = (int32_t)mxGetScalar(prhs[7]),
        .bs = (int32_t)mxGetScalar(prhs[8]),
        .l = (int32_t)mxGetScalar(prhs[9]),
        .tol = (double)mxGetScalar(prhs[3])
    };

    if (iter.l > MAX_DEPTH) {
        mexErrMsgTxt("nlevels too large. Increase MAX_DEPTH.");
    }

    struct MG mg[MAX_DEPTH];

    mx_init(mg, prhs[0], prhs[1], prhs[2], iter.l);
    mx_fmg(mg, &iter, 0);

    plhs[0] = mx_unpad_boundary(mg[0].v);

    mx_cleanup(mg, iter.l);

    return;
}


void
mx_fmg(struct MG *mg, const struct Iter *iter, const int32_t l)
{
    if (l < iter->l-1) {
        mx_fmg(mg, iter, l+1);
        mx_prolong(mg[l].v, mg[l+1].v, mg[l].G);
    }

    if ((l == 0) && (iter->tol > 0.0)) {
        const double reltol = iter->tol * mx_nrm2(mg[0].f);

        /* TODO: maxit input param */
        for(int32_t i = 0; i < 1000*iter->mg; ++i) {
            mx_residual(mg[0].r, mg[0].f, mg[0].v, mg[0].G, mg[0].h);
            if (mx_nrm2(mg[0].r) < reltol) break;

            mx_cycle(mg, iter, l);
        }

    } else {
        for(int32_t i = 0; i < iter->mg; ++i) {
            mx_cycle(mg, iter, l);
        }
    }

    return;
}


void
mx_cycle(struct MG *mg, const struct Iter *iter, const int32_t l)
{
    if (l < iter->l-1) {
        const int32_t bs = (1<<l) * iter->bs;

        mx_gauss_seidel(mg[l].v, mg[l].f, mg[l].G, mg[l].h, iter->pre, 1);
        mx_gauss_seidel(mg[l].v, mg[l].f, mg[l].B, mg[l].h, bs, 1);

        mx_residual(mg[l].r, mg[l].f, mg[l].v, mg[l].G, mg[l].h);
        mx_restrict(mg[l+1].f, mg[l].r, mg[l+1].G);

        mx_zero(mg[l+1].v);
        for(int32_t i = 0; i < iter->mu; ++i) {
            mx_cycle(mg, iter, l+1);
            if (l+1 == iter->l-1) break;
        }

        mx_correct(mg[l].v, mg[l+1].v, mg[l].G);

        mx_gauss_seidel(mg[l].v, mg[l].f, mg[l].B, mg[l].h, bs, 0);
        mx_gauss_seidel(mg[l].v, mg[l].f, mg[l].G, mg[l].h, iter->post, 0);

    } else {
        const double tol = mxIsSingle(mg[l].f) ? 1e-4 : 1e-8;
        const double reltol = tol * mx_nrm2(mg[l].f);

        for(int32_t i = 0; i < 1024; ++i) {
            mx_gauss_seidel(mg[l].v, mg[l].f, mg[l].G, mg[l].h, 128, 1);
            mx_gauss_seidel(mg[l].v, mg[l].f, mg[l].G, mg[l].h, 128, 0);

            mx_residual(mg[l].r, mg[l].f, mg[l].v, mg[l].G, mg[l].h);
            if (mx_nrm2(mg[l].r) < reltol) break;
        }
    }

    return;
}


void
mx_init(struct MG *mg,
        const mxArray *mxf, const mxArray *mxG,
        const mxArray *mxh, const int32_t l)
{
    double *h = NULL;

    mxArray *mxfp = mx_pad_boundary(mxf);
    mxArray *mxGp = mx_pad_boundary(mxG);

    mxClassID T = mxGetClassID(mxfp);

    const size_t *szp = (const size_t *)mxGetDimensions(mxfp);
    size_t sz[3] = {szp[0], szp[1], szp[2]};

    mg[0].f = mxfp;
    mg[0].G = mxGp;
    mg[0].h = mxDuplicateArray(mxh);
    mg[0].v = mxCreateNumericArray(3, sz, T, mxREAL);
    mg[0].r = mxCreateNumericArray(3, sz, T, mxREAL);
    mg[0].B = mxCreateNumericArray(3, sz, mxGetClassID(mxG), mxREAL);

    mx_boundary_mask(mg[0].B, mg[0].G);

    for(int32_t i = 1; i < l; ++i) {
        sz[0] = ((sz[0]-2+1)>>1) + 2;
        sz[1] = ((sz[1]-2+1)>>1) + 2;
        sz[2] = ((sz[2]-2+1)>>1) + 2;

        mg[i].v = mxCreateNumericArray(3, sz, T, mxREAL);
        mg[i].r = mxCreateNumericArray(3, sz, T, mxREAL);
        mg[i].f = mxCreateNumericArray(3, sz, T, mxREAL);
        mg[i].G = mxCreateNumericArray(3, sz, mxGetClassID(mxG), mxREAL);
        mg[i].B = mxCreateNumericArray(3, sz, mxGetClassID(mxG), mxREAL);

        mx_coarsen_grid(mg[i].G, mg[i-1].G);
        mx_boundary_mask(mg[i].B, mg[i].G);
        mx_restrict(mg[i].f, mg[i-1].f, mg[i].G);

        mg[i].h = mxDuplicateArray(mg[i-1].h);
        h = (double *)mxGetData(mg[i].h);

        h[0] *= 2.0;
        h[1] *= 2.0;
        h[2] *= 2.0;
    }

    h = NULL;

    return;
}


void
mx_cleanup(struct MG *mg, const int32_t l)
{
    for(int32_t i = 0; i < l; ++i) {
        if (NULL != mg[i].v) {
            mxDestroyArray(mg[i].v);
            mg[i].v = NULL;
        }
        if (NULL != mg[i].f) {
            mxDestroyArray(mg[i].f);
            mg[i].f = NULL;
        }
        if (NULL != mg[i].r) {
            mxDestroyArray(mg[i].r);
            mg[i].r = NULL;
        }
        if (NULL != mg[i].G) {
            mxDestroyArray(mg[i].G);
            mg[i].G = NULL;
        }
        if (NULL != mg[i].B) {
            mxDestroyArray(mg[i].B);
            mg[i].B = NULL;
        }
        if (NULL != mg[i].h) {
            mxDestroyArray(mg[i].h);
            mg[i].h = NULL;
        }
    }
    return;
}
