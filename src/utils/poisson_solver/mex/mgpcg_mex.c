#include <inttypes.h>
#include "mex.h"

#include "boundary_mask_mex.h"
#include "coarsen_grid_mex.h"
#include "correct_mex.h"
#include "gauss_seidel_mex.h"
#include "lapmg_mex.h"
#include "mx_blas.h"
#include "mx_util.h"
#include "prolong_mex.h"
#include "restrict_mex.h"
#include "residual_mex.h"


#define MAX_DEPTH 16


/* TODO: return convergence history struct */
struct PCG
{
    uint8_t  isconverged;
    int32_t  iters;
    mxArray *resnorm;
    double   reltol;
    const double tol;
    const int32_t maxit;
};


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
             const mxArray *mxv,
             const mxArray *mxf,
             const mxArray *mxG,
             const mxArray *mxh,
             const int32_t l);

void mx_pcg(mxArray *x,
            struct PCG *ch,
            struct MG *mg,
            const struct Iter *iter,
            const uint8_t verbose);

void mx_mg(struct MG *mg, const struct Iter *iter);
void mx_cycle(struct MG *mg, const struct Iter *iter, const int32_t l);

void mx_cleanup(mxArray *x, struct MG *mg, const int32_t l);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 14) || (nlhs > 1)) {
        mexErrMsgTxt(
            "Usage: v = mgpcg_mex("
                "v, f, mask, vsz, "
                "tolcg, maxitcg, "
                "tolmg, maxitmg, mu, npre, npost, nboundary, nlevels, verbose)"
        );
        return;
    }

    struct PCG ch = {
        .isconverged = 0,
        .iters = 0,
        .resnorm = NULL,
        .tol = (double)mxGetScalar(prhs[4]),
        .maxit = (int32_t)mxGetScalar(prhs[5])
    };

    ch.resnorm = mxCreateDoubleMatrix(ch.maxit, 1, mxREAL);

    const struct Iter iter = {
        .mg   = (int32_t)mxGetScalar(prhs[7]),
        .mu   = (int32_t)mxGetScalar(prhs[8]),
        .pre  = (int32_t)mxGetScalar(prhs[9]),
        .post = (int32_t)mxGetScalar(prhs[10]),
        .bs   = (int32_t)mxGetScalar(prhs[11]),
        .l    = (int32_t)mxGetScalar(prhs[12]),
        .tol  = (double)mxGetScalar(prhs[6]),
    };

    const uint8_t verbose = (uint8_t)mxGetScalar(prhs[13]);

    if (iter.l > MAX_DEPTH) {
        mexErrMsgTxt("nlevels too large. Increase MAX_DEPTH.");
    }

    struct MG mg[MAX_DEPTH];

    mx_init(mg, prhs[0], prhs[1], prhs[2], prhs[3], iter.l);
    mxArray *x = mxDuplicateArray(mg[0].v); /* initial guess */
    mx_pcg(x, &ch, mg, &iter, verbose);

    plhs[0] = mx_unpad_boundary(x);

    mx_cleanup(x, mg, iter.l);

    return;
}


void
mx_pcg(mxArray *x, struct PCG *ch, struct MG *mg,
       const struct Iter *iter, const uint8_t verbose)
{
    const double ONE = 1.0;

    double *resnorm = (double *)mxGetData(ch->resnorm);

    /* initialize pcg scalars */
    double a = 1.0;
    double beta = 1.0;
    double rho = 1.0;
    double prevrho = rho;

    double res = mx_nrm2(mg[0].f);
    const double reltol = ch->tol * res;

    ch->reltol = reltol;

    /* get arrays used in pcg */
    mxArray *G = mg[0].G;
    mxArray *h = mg[0].h;

    /* reusing mg variables for pcg - q = M\r */
    mxArray *r = mg[0].f;
    mxArray *q = mg[0].v;
    mxArray *p = mxDuplicateArray(q);

    mx_zero(q);
    mx_zero(p);

    /* initial guess */
    mx_lapmg(q, x, G, h);
    mx_scal(-ONE, q);       /* q = Ax; */
    mx_axpy(-ONE, q, r);    /* r = r - q; */
    res = mx_nrm2(r);

    if (verbose) {
        mexPrintf("\niter%6s\tresidual\n", "");
    }

    for(int32_t i = 0; i < ch->maxit; ++i) {
        if (res <= reltol) {
            ch->isconverged = 1;
            break;
        }

        prevrho = rho;

        mx_zero(q);
        mx_mg(mg, iter);        /* q = M\r */

        rho = mx_dot(r, q);     /* rho = r'.q */
        beta = rho / prevrho;

        mx_scal(beta, p);
        mx_axpy(ONE, q, p);     /* p = q + beta * p */

        mx_lapmg(q, p, G, h);
        mx_scal(-ONE, q);       /* q = Ap */
        a = rho / mx_dot(p, q); /* a = r'.q / p'.Ap */

        mx_axpy(a, p, x);       /* x = x + a * p */
        mx_axpy(-a, q, r);      /* r = r - a * q */

        res = mx_nrm2(r);

        resnorm[i] = res;
        ch->iters += 1;

        if (verbose) {
            mexPrintf("%5d/%d\t%1.3e\n", i, ch->maxit, res);
        }
    }

    if (verbose) {
        mexPrintf("\n");
    }

    if (NULL != p) {
        mxDestroyArray(p);
        p = NULL;
    }

    r = NULL;
    q = NULL;
    G = NULL;
    h = NULL;

    return;
}


void
mx_mg(struct MG *mg, const struct Iter *iter)
{
    if (iter->tol > 0.0) {
        const double reltol = iter->tol * mx_nrm2(mg[0].f);

        for(int32_t i = 0; i < iter->mg; ++i) {
            mx_residual(mg[0].r, mg[0].f, mg[0].v, mg[0].G, mg[0].h);
            if (mx_nrm2(mg[0].r) < reltol) break;

            mx_cycle(mg, iter, 0);
        }

    } else {
        for(int32_t i = 0; i < iter->mg; ++i) {
            mx_cycle(mg, iter, 0);
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
        const mxArray *mxv, const mxArray *mxf, const mxArray *mxG,
        const mxArray *mxh, const int32_t l)
{
    double *h = NULL;

    mxArray *mxvp = mx_pad_boundary(mxv);
    mxArray *mxfp = mx_pad_boundary(mxf);
    mxArray *mxGp = mx_pad_boundary(mxG);

    mxClassID T = mxGetClassID(mxvp);

    const size_t *szp = (const size_t *)mxGetDimensions(mxvp);
    size_t sz[3] = {szp[0], szp[1], szp[2]};

    mg[0].v = mxvp;
    mg[0].f = mxfp;
    mg[0].G = mxGp;
    mg[0].h = mxDuplicateArray(mxh);
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
mx_cleanup(mxArray *x, struct MG *mg, const int32_t l)
{
    if (NULL != x) {
        mxDestroyArray(x);
        x = NULL;
    }

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
