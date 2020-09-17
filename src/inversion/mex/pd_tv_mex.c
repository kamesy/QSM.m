#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include "mex.h"


#define MIN_VERBOSE (4)
#define CONVERGENCE_MOD (10)

/* preprocessor can't compare floating point numbers: */
/*  THETA0 = 1 -> THETA = 0.0   */
/*  THETA0 = 0 -> THETA = THETA */
#define THETA0 (0)
#define THETA (1.0)

#define OI (1 << 1)             /* unused */
#define BI (1 << 2)             /* unused */
#define FI (1 << 3)             /* unused */
#define CI (1 << 4)             /* unused */

#define OJ (1 << 5)             /* unused */
#define BJ (1 << 6)             /* unused */
#define FJ (1 << 7)             /* unused */
#define CJ (1 << 8)             /* unused */

#define OK (1 << 9)             /* unused */
#define BK (1 << 10)            /* unused */
#define FK (1 << 11)            /* unused */
#define CK (1 << 12)            /* unused */

#define INSIDE (CI + CJ + CK)
#define OUTSIDE (OI + OJ + OK)

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))


struct PD
{
    /* primal */
    void *chi;
    void *eta;
    void *chi_;
    void *eta_;
    /* dual */
    void *nu;
    void *p;
    /* misc */
    void *f;
    void *m;
    void *mi;
    void *mb;       /* unused */
    void *chi0;
    void *eta0;
};


void mx_pdd(struct PD *mxpd,
            const double lam, const double *h,
            const double tol, const uint32_t maxiter, const uint32_t verbose);

void mx_pdf(struct PD *mxpd,
            const double lam, const double *h,
            const double tol, const uint32_t maxiter, const uint32_t verbose);


void update_duald(struct PD *pd,
                  const double lam, const double sigma,
                  const double *h, const size_t *sz);

void update_dualf(struct PD *pd,
                  const double lam, const double sigma,
                  const double *h, const size_t *sz);


void update_primald(struct PD *pd,
                    const double tau, const double *h, const size_t *sz);

void update_primalf(struct PD *pd,
                    const double tau, const double *h, const size_t *sz);


uint8_t convergence_checkd(double *nr1, double *nr2,
                           const double *chi, const double *chi0,
                           const double *eta, const double *eta0,
                           const double tol, const size_t N);

uint8_t convergence_checkf(double *nr1, double *nr2,
                           const float *chi, const float *chi0,
                           const float *eta, const float *eta0,
                           const double tol, const size_t N);


void init_bitmask(uint32_t *mb, uint8_t *mi, const uint8_t *m, const size_t *sz);

void mx_init(struct PD *pd, const mxArray *mxf, const mxArray *mxm);
void mx_cleanup(struct PD *pd);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 7) || (nlhs > 1)) {
        mexErrMsgTxt(
            "Usage: chi = pd_tv_mex(d2f, mask, vsz, lam, tol, maxiter, verbose);"
        );
        return;
    }

    const double *h = (const double *)mxGetData(prhs[2]);
    const double lam = (const double)mxGetScalar(prhs[3]);
    const double tol = (const double)mxGetScalar(prhs[4]);
    const uint32_t maxiter = (const uint32_t)mxGetScalar(prhs[5]);
    const uint32_t verbose = (const uint32_t)mxGetScalar(prhs[6]);

    struct PD mxpd;
    mx_init(&mxpd, prhs[0], prhs[1]);

    if (mxIsSingle(prhs[0])) {
        mx_pdf(&mxpd, lam, h, tol, maxiter, verbose);
    } else {
        mx_pdd(&mxpd, lam, h, tol, maxiter, verbose);
    }

    plhs[0] = mxDuplicateArray(mxpd.chi);

    mx_cleanup(&mxpd);

    return;
}


void
mx_pdf(struct PD *mxpd,
       const double lam, const double *h,
       const double tol, const uint32_t maxiter, const uint32_t verbose)
{
    uint32_t i, l, pmod;
    double nr1, nr2;

    const size_t N = (const size_t)mxGetNumberOfElements(mxpd->f);
    const size_t *sz = (const size_t *)mxGetDimensions(mxpd->f);

    struct PD pd;

    pd.chi  = (float *)mxGetData(mxpd->chi);
    pd.eta  = (float *)mxGetData(mxpd->eta);
    pd.chi_ = (float *)mxGetData(mxpd->chi_);
    pd.eta_ = (float *)mxGetData(mxpd->eta_);

    pd.nu = (float *)mxGetData(mxpd->nu);
    pd.p = (float *)mxGetData(mxpd->p);

    pd.f = (float *)mxGetData(mxpd->f);
    pd.m = (uint8_t *)mxGetData(mxpd->m);
    pd.mi = (uint8_t *)mxGetData(mxpd->mi);
    pd.mb = (uint32_t *)mxGetData(mxpd->mb);
    pd.chi0 = (float *)mxGetData(mxpd->chi0);
    pd.eta0 = (float *)mxGetData(mxpd->eta0);

    float *f = pd.f;
    float *chi = pd.chi;
    float *eta = pd.eta;
    float *chi0 = pd.chi0;
    float *eta0 = pd.eta0;

    double gnrm = 4.0 * (1.0/(h[0]*h[0]) + 1.0/(h[1]*h[1]) + 1.0/(h[2]*h[2]));
    double wnrm = 6.0 * (1.0/(h[0]*h[0]) + 1.0/(h[1]*h[1]) + 2.0/(h[2]*h[2])) / 3.0;

    double K2 = gnrm*gnrm + wnrm*wnrm;

    const double tau = 1/sqrt(K2);
    const double sigma = 1/(tau*K2);

    #pragma omp parallel for private(l) schedule(static)
    for(l = 0; l < N; l += 1) {
        f[l] = ((float)sigma)*f[l];
    }

    uint8_t flag = 0;

    nr1 = 1.0;
    nr2 = 1.0;

    pmod = maxiter / MIN(MAX(verbose, MIN_VERBOSE), maxiter);

    if (verbose) {
        mexPrintf("Iter\t\t||delta chi||\t||delta eta||\n");
    }

    for(i = 0; i < maxiter; ++i) {

        if (((i+1) % CONVERGENCE_MOD) == 0) {
            #pragma omp parallel for private(l) schedule(static)
            for(l = 0; l < N; l += 1) {
                chi0[l] = chi[l];
                eta0[l] = eta[l];
            }
        }

        update_dualf(&pd, lam, sigma, h, sz);
        update_primalf(&pd, tau, h, sz);

        if (((i+1) % CONVERGENCE_MOD) == 0) {
            flag = convergence_checkf(&nr1, &nr2, chi, chi0, eta, eta0, tol, N);
        }

        if (verbose && (i == 0 || i == maxiter-1 || ((i+1) % pmod) == 0 || flag)) {
            mexPrintf("%5d/%d\t%.3e\t%.3e\n", i+1, maxiter, nr1, nr2);
        }

        if (flag) {
            break;
        }
    }

    return;
}


void
mx_pdd(struct PD *mxpd,
       const double lam, const double *h,
       const double tol, const uint32_t maxiter, const uint32_t verbose)
{
    uint32_t i, l, pmod;
    double nr1, nr2;

    const size_t N = (const size_t)mxGetNumberOfElements(mxpd->f);
    const size_t *sz = (const size_t *)mxGetDimensions(mxpd->f);

    struct PD pd;

    pd.chi  = (double *)mxGetData(mxpd->chi);
    pd.eta  = (double *)mxGetData(mxpd->eta);
    pd.chi_ = (double *)mxGetData(mxpd->chi_);
    pd.eta_ = (double *)mxGetData(mxpd->eta_);

    pd.nu = (double *)mxGetData(mxpd->nu);
    pd.p = (double *)mxGetData(mxpd->p);

    pd.f = (double *)mxGetData(mxpd->f);
    pd.m = (uint8_t *)mxGetData(mxpd->m);
    pd.mi = (uint8_t *)mxGetData(mxpd->mi);
    pd.mb = (uint32_t *)mxGetData(mxpd->mb);
    pd.chi0 = (double *)mxGetData(mxpd->chi0);
    pd.eta0 = (double *)mxGetData(mxpd->eta0);

    double *f = pd.f;
    double *chi = pd.chi;
    double *eta = pd.eta;
    double *chi0 = pd.chi0;
    double *eta0 = pd.eta0;

    double gnrm = 4.0 * (1.0/(h[0]*h[0]) + 1.0/(h[1]*h[1]) + 1.0/(h[2]*h[2]));
    double wnrm = 6.0 * (1.0/(h[0]*h[0]) + 1.0/(h[1]*h[1]) + 2.0/(h[2]*h[2])) / 3.0;

    double K2 = gnrm*gnrm + wnrm*wnrm;

    const double tau = 1/sqrt(K2);
    const double sigma = 1/(tau*K2);

    #pragma omp parallel for private(l) schedule(static)
    for(l = 0; l < N; l += 1) {
        f[l] = sigma*f[l];
    }

    uint8_t flag = 0;

    nr1 = 1.0;
    nr2 = 1.0;

    pmod = maxiter / MIN(MAX(verbose, MIN_VERBOSE), maxiter);

    if (verbose) {
        mexPrintf("Iter\t\t||delta chi||\t||delta eta||\n");
    }

    for(i = 0; i < maxiter; ++i) {

        if (((i+1) % CONVERGENCE_MOD) == 0) {
            #pragma omp parallel for private(l) schedule(static)
            for(l = 0; l < N; l += 1) {
                chi0[l] = chi[l];
                eta0[l] = eta[l];
            }
        }

        update_duald(&pd, lam, sigma, h, sz);
        update_primald(&pd, tau, h, sz);

        if (((i+1) % CONVERGENCE_MOD) == 0) {
            flag = convergence_checkd(&nr1, &nr2, chi, chi0, eta, eta0, tol, N);
        }

        if (verbose && (i == 0 || i == maxiter-1 || ((i+1) % pmod) == 0 || flag)) {
            mexPrintf("%5d/%d\t%.3e\t%.3e\n", i+1, maxiter, nr1, nr2);
        }

        if (flag) {
            break;
        }
    }

    return;
}


/*
 * nu <- nu + sigma*(laplace(eta_) - wave(chi_) + f)
 * p <- P_{||.||_inf <= lam}(p + sigma*grad(chi_))
 */
void
update_dualf(struct PD *pd,
             const double lam, const double sigma,
             const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    float x;

    float *nu = (float *)pd->nu;
    float *p = (float *)pd->p;

    #if !THETA0
    float *chi_ = (float *)pd->chi_;
    float *eta_ = (float *)pd->eta_;
    #else
    float *chi_ = (float *)pd->chi;
    float *eta_ = (float *)pd->eta;
    #endif

    float *f = (float *)pd->f;
    uint8_t *m = (uint8_t *)pd->mi;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const float hx = (float)(sigma/h[0]);
    const float hy = (float)(sigma/h[1]);
    const float hz = (float)(sigma/h[2]);

    const float lhx = (float)(sigma/(h[0]*h[0]));
    const float lhy = (float)(sigma/(h[1]*h[1]));
    const float lhz = (float)(sigma/(h[2]*h[2]));
    const float lhh = -2.0f*(lhx+lhy+lhz);

    const float whx = (float)(-sigma/(3.0*h[0]*h[0]));
    const float why = (float)(-sigma/(3.0*h[1]*h[1]));
    const float whz = (float)( sigma*2.0/(3.0*h[2]*h[2]));
    const float whh = -2.0f*(whx+why+whz);

    const float lam_ = (float)lam;
    const float lam2_ = (float)(lam*lam);

    float *p1 = &p[0*(nxny*nz)];
    float *p2 = &p[1*(nxny*nz)];
    float *p3 = &p[2*(nxny*nz)];


    #pragma omp parallel for private(i,j,k,l,x) schedule(static) \
        if (nxny*nz > 32*32*32)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                if (m[l] != 0) {
                    nu[l] = nu[l] + (
                    /* laplace(eta_) */
                        (lhh*eta_[l] +
                        lhx*(eta_[l-1] + eta_[l+1]) +
                        lhy*(eta_[l-nx] + eta_[l+nx]) +
                        lhz*(eta_[l-nxny] + eta_[l+nxny])) -
                    /* - wave(chi_) */
                        (whh*chi_[l] +
                        whx*(chi_[l-1] + chi_[l+1]) +
                        why*(chi_[l-nx] + chi_[l+nx]) +
                        whz*(chi_[l-nxny] + chi_[l+nxny])) +
                    /* + f */
                        f[l]
                    );

                    /* p + sigma*grad(chi_) */
                    p1[l] = p1[l] + hx*(chi_[l+1]-chi_[l]);
                    p2[l] = p2[l] + hy*(chi_[l+nx]-chi_[l]);
                    p3[l] = p3[l] + hz*(chi_[l+nxny]-chi_[l]);

                    /* p <- P_{||.||_inf <= lam}(p) */
                    x = p1[l]*p1[l] + p2[l]*p2[l] + p3[l]*p3[l];

                    if (x > lam2_) {
                        x = lam_/sqrtf(x);
                        p1[l] = p1[l] * x;
                        p2[l] = p2[l] * x;
                        p3[l] = p3[l] * x;
                    }
                }
            }
        }
    }

    return;
}


/*
 * nu <- nu + sigma*(laplace(eta_) - wave(chi_) + f)
 * p <- P_{||.||_inf <= lam}(p + sigma*grad(chi_))
 */
void
update_duald(struct PD *pd,
             const double lam, const double sigma,
             const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    double x;

    double *nu = (double *)pd->nu;
    double *p = (double *)pd->p;

    #if !THETA0
    double *chi_ = (double *)pd->chi_;
    double *eta_ = (double *)pd->eta_;
    #else
    double *chi_ = (double *)pd->chi;
    double *eta_ = (double *)pd->eta;
    #endif

    double *f = (double *)pd->f;
    uint8_t *m = (uint8_t *)pd->mi;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const double hx = sigma/h[0];
    const double hy = sigma/h[1];
    const double hz = sigma/h[2];

    const double lhx =  sigma/(h[0]*h[0]);
    const double lhy =  sigma/(h[1]*h[1]);
    const double lhz =  sigma/(h[2]*h[2]);
    const double lhh = -2.0*(lhx+lhy+lhz);

    const double whx = -sigma/(3.0*h[0]*h[0]);
    const double why = -sigma/(3.0*h[1]*h[1]);
    const double whz =  sigma*2.0/(3.0*h[2]*h[2]);
    const double whh = -2.0*(whx+why+whz);

    const double lam2 = lam*lam;

    double *p1 = &p[0*(nxny*nz)];
    double *p2 = &p[1*(nxny*nz)];
    double *p3 = &p[2*(nxny*nz)];


    #pragma omp parallel for private(i,j,k,l,x) schedule(static) \
        if (nxny*nz > 32*32*32)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                if (m[l] != 0) {
                    nu[l] = nu[l] + (
                    /* laplace(eta_) */
                        (lhh*eta_[l] +
                        lhx*(eta_[l-1] + eta_[l+1]) +
                        lhy*(eta_[l-nx] + eta_[l+nx]) +
                        lhz*(eta_[l-nxny] + eta_[l+nxny])) -
                    /* - wave(chi_) */
                        (whh*chi_[l] +
                        whx*(chi_[l-1] + chi_[l+1]) +
                        why*(chi_[l-nx] + chi_[l+nx]) +
                        whz*(chi_[l-nxny] + chi_[l+nxny])) +
                    /* + f */
                        f[l]
                    );

                    /* p + sigma*grad(chi_) */
                    p1[l] = p1[l] + hx*(chi_[l+1]-chi_[l]);
                    p2[l] = p2[l] + hy*(chi_[l+nx]-chi_[l]);
                    p3[l] = p3[l] + hz*(chi_[l+nxny]-chi_[l]);

                    /* p <- P_{||.||_inf <= lam}(p) */
                    x = p1[l]*p1[l] + p2[l]*p2[l] + p3[l]*p3[l];

                    if (x > lam2) {
                        x = lam/sqrt(x);
                        p1[l] = p1[l] * x;
                        p2[l] = p2[l] * x;
                        p3[l] = p3[l] * x;
                    }
                }
            }
        }
    }

    return;
}


/*
 * eta  <- (eta - tau*laplace(nu)) / (1 + tau)
 * chi  <-  chi - tau*(-div(p) - wave(nu))
 *
 * eta_ <-  eta_n+1 + THETA*(eta_n+1 - eta_n)
 * chi_ <-  chi_n+1 + THETA*(chi_n+1 - chi_n)
 */
void
update_primalf(struct PD *pd,
               const double tau, const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    float x, y, z;

    float *chi  = (float *)pd->chi;
    float *eta  = (float *)pd->eta;
    #if !THETA0
    float cn, en;
    float *chi_ = (float *)pd->chi_;
    float *eta_ = (float *)pd->eta_;
    #endif

    float *nu = (float *)pd->nu;
    float *p = (float *)pd->p;

    uint8_t *m = (uint8_t *)pd->m;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const float hx = (float)(tau/h[0]);
    const float hy = (float)(tau/h[1]);
    const float hz = (float)(tau/h[2]);

    const float lhx = (float)(tau/(h[0]*h[0]));
    const float lhy = (float)(tau/(h[1]*h[1]));
    const float lhz = (float)(tau/(h[2]*h[2]));
    const float lhh = -2.0f*(lhx+lhy+lhz);

    const float whx = (float)(-tau/(3.0*h[0]*h[0]));
    const float why = (float)(-tau/(3.0*h[1]*h[1]));
    const float whz = (float)( tau*2.0/(3.0*h[2]*h[2]));
    const float whh = -2.0f*(whx+why+whz);

    const float tau1 = (float)(1.0 / (1.0 + tau));

    float *p1 = &p[0*(nxny*nz)];
    float *p2 = &p[1*(nxny*nz)];
    float *p3 = &p[2*(nxny*nz)];


    #pragma omp parallel for private(i,j,k,l,x,y,z) schedule(static) \
        if (nxny*nz > 32*32*32)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                if (m[l] != 0) {
                    /* eta_ <- eta_n+1 + theta*(eta_n+1 - eta_n) */
                    /* chi_ <- chi_n+1 + theta*(chi_n+1 - chi_n) */
                    #if !THETA0
                    en = eta[l];
                    cn = chi[l];
                    #endif

                    x = (nu[l-1] + nu[l+1]);
                    y = (nu[l-nx] + nu[l+nx]);
                    z = (nu[l-nxny] + nu[l+nxny]);

                    /* eta_n+1 <- (eta_n - tau*laplace(nu)) / (1+tau) */
                    eta[l] = tau1 * (
                        eta[l] - lhh*nu[l] - lhx*x - lhy*y - lhz*z
                    );

                    /* chi_n+1 <- chi_n - tau*(-div(p) - wave(nu)) */
                    chi[l] = chi[l] + (
                        /* div(p) */
                        hx*(p1[l]-p1[l-1]) +
                        hy*(p2[l]-p2[l-nx]) +
                        hz*(p3[l]-p3[l-nxny]) +
                        /* wave(nu) */
                        (whh*nu[l] + whx*x + why*y + whz*z)
                    );

                    /* eta_ <- eta_n+1 + theta*(eta_n+1 - eta_n) */
                    /* chi_ <- chi_n+1 + theta*(chi_n+1 - chi_n) */
                    #if !THETA0
                    eta_[l] = eta[l] + ((float)THETA)*(eta[l] - en);
                    chi_[l] = chi[l] + ((float)THETA)*(chi[l] - cn);
                    #endif
                }
            }
        }
    }

    return;
}


/*
 * eta  <- (eta - tau*laplace(nu)) / (1 + tau)
 * chi  <-  chi - tau*(-div(p) - wave(nu))
 *
 * eta_ <-  eta_n+1 + THETA*(eta_n+1 - eta_n)
 * chi_ <-  chi_n+1 + THETA*(chi_n+1 - chi_n)
 */
void
update_primald(struct PD *pd,
               const double tau, const double *h, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    double x, y, z;

    double *chi  = (double *)pd->chi;
    double *eta  = (double *)pd->eta;
    #if !THETA0
    double cn, en;
    double *chi_ = (double *)pd->chi_;
    double *eta_ = (double *)pd->eta_;
    #endif

    double *nu = (double *)pd->nu;
    double *p  = (double *)pd->p;

    uint8_t *m = (uint8_t *)pd->m;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    const double hx = tau/h[0];
    const double hy = tau/h[1];
    const double hz = tau/h[2];

    const double lhx =  tau/(h[0]*h[0]);
    const double lhy =  tau/(h[1]*h[1]);
    const double lhz =  tau/(h[2]*h[2]);
    const double lhh = -2.0*(lhx+lhy+lhz);

    const double whx = -tau/(3.0*h[0]*h[0]);
    const double why = -tau/(3.0*h[1]*h[1]);
    const double whz =  tau*2.0/(3.0*h[2]*h[2]);
    const double whh = -2.0*(whx+why+whz);

    const double tau1 = 1.0 / (1.0 + tau);

    double *p1 = &p[0*(nxny*nz)];
    double *p2 = &p[1*(nxny*nz)];
    double *p3 = &p[2*(nxny*nz)];


    #pragma omp parallel for private(i,j,k,l,x,y,z) schedule(static) \
        if (nxny*nz > 32*32*32)
    for(k = nxny; k < NZ; k += nxny) {
        for(j = nx; j < NY; j += nx) {
            l = 1 + j + k;
            for(i = 1; i < NX; ++i, ++l) {
                if (m[l] != 0) {
                    /* eta_ <- eta_n+1 + theta*(eta_n+1 - eta_n) */
                    /* chi_ <- chi_n+1 + theta*(chi_n+1 - chi_n) */
                    #if !THETA0
                    en = eta[l];
                    cn = chi[l];
                    #endif

                    x = (nu[l-1] + nu[l+1]);
                    y = (nu[l-nx] + nu[l+nx]);
                    z = (nu[l-nxny] + nu[l+nxny]);

                    /* eta_n+1 <- (eta_n - tau*laplace(nu)) / (1+tau) */
                    eta[l] = tau1 * (
                        eta[l] - lhh*nu[l] - lhx*x - lhy*y - lhz*z
                    );

                    /* chi_n+1 <- chi_n - tau*(-div(p) - wave(nu)) */
                    chi[l] = chi[l] + (
                        /* div(p) */
                        hx*(p1[l]-p1[l-1]) +
                        hy*(p2[l]-p2[l-nx]) +
                        hz*(p3[l]-p3[l-nxny]) +
                        /* wave(nu) */
                        (whh*nu[l] + whx*x + why*y + whz*z)
                    );

                    /* eta_ <- eta_n+1 + theta*(eta_n+1 - eta_n) */
                    /* chi_ <- chi_n+1 + theta*(chi_n+1 - chi_n) */
                    #if !THETA0
                    eta_[l] = eta[l] + ((double)THETA)*(eta[l] - en);
                    chi_[l] = chi[l] + ((double)THETA)*(chi[l] - cn);
                    #endif
                }
            }
        }
    }

    return;
}


uint8_t
convergence_checkd(double *nr1, double *nr2,
                   const double *chi, const double *chi0,
                   const double *eta, const double *eta0,
                   const double tol, const size_t N)
{
    size_t l;

    double n1 = 0.0;
    double n2 = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;

    /* can do naive summation for this */
    /* pairwise summation norm is with the multigrid stuff */
    #pragma omp parallel for private(l) \
        reduction(+: n1, n2, d1, d2) schedule(static)
    for(l = 0; l < N; l += 1) {
        d1 += chi[l]*chi[l];
        d2 += eta[l]*eta[l];
        n1 += (chi[l]-chi0[l]) * (chi[l]-chi0[l]);
        n2 += (eta[l]-eta0[l]) * (eta[l]-eta0[l]);
    }

    n1 = sqrt(n1/d1);
    n2 = sqrt(n2/d2);

    *nr1 = n1;
    *nr2 = n2;

    return (uint8_t)((n1 < tol) && (n2 < 2.0*tol));
}


uint8_t
convergence_checkf(double *nr1, double *nr2,
                   const float *chi, const float *chi0,
                   const float *eta, const float *eta0,
                   const double tol, const size_t N)
{
    size_t l;

    double n1 = 0.0;
    double n2 = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;

    /* can do naive summation for this */
    /* pairwise summation norm is with the multigrid stuff */
    #pragma omp parallel for private(l) \
        reduction(+: n1, n2, d1, d2) schedule(static)
    for(l = 0; l < N; l += 1) {
        d1 += ((double)chi[l]) * ((double)chi[l]);
        d2 += ((double)eta[l]) * ((double)eta[l]);
        n1 += ((double)(chi[l]-chi0[l])) * ((double)chi[l]-chi0[l]);
        n2 += ((double)(eta[l]-eta0[l])) * ((double)eta[l]-eta0[l]);
    }

    n1 = sqrt(n1/d1);
    n2 = sqrt(n2/d2);

    *nr1 = n1;
    *nr2 = n2;

    return (uint8_t)((n1 < tol) && (n2 < 2.0*tol));
}



void
init_bitmask(uint32_t *mb, uint8_t *mi, const uint8_t *m, const size_t *sz)
{
    size_t i, j, k;
    size_t l;

    uint8_t b, f, o;

    const size_t nx = sz[0];
    const size_t ny = sz[1];
    const size_t nz = sz[2];
    const size_t nxny = nx*ny;

    const size_t NX = nx-1;
    const size_t NY = nx*(ny-1);
    const size_t NZ = nxny*(nz-1);

    /*
     *     0  b  f  c
     * --------------
     * i   1  2  3  4
     * j   5  6  7  8
     * k   9 10 11 12
     */
    #pragma omp parallel for private(i,j,k,l,b,f,o) schedule(static) \
        if (nxny*nz > 32*32*32)
    for(k = 0; k <= NZ; k += nxny) {
        for(j = 0; j <= NY; j += nx) {
            l = j + k;
            for(i = 0; i <= NX; ++i, ++l) {
                o = 3;
                f = (i < NX) && m[l+1];
                mb[l] |= f ? (1 << o) : 0;

                o = 2;
                b = (i > 0) && m[l-1];
                mb[l] |= b ? (1 << o) : 0;

                o = 1;
                mb[l] |= (1 << (o + 2*f + b));


                o = 7;
                f = (j < NY) && m[l+nx];
                mb[l] |= f ? (1 << o) : 0;

                o = 6;
                b = (j > 0) && m[l-nx];
                mb[l] |= b ? (1 << o) : 0;

                o = 5;
                mb[l] |= (1 << (o + 2*f + b));


                o = 11;
                f = (k < NZ) && m[l+nxny];
                mb[l] |= f ? (1 << o) : 0;

                o = 10;
                b = (k > 0) && m[l-nxny];
                mb[l] |= b ? (1 << o) : 0;

                o = 9;
                mb[l] |= (1 << (o + 2*f + b));

                mb[l] = ((mb[l] & OUTSIDE) == OUTSIDE) ? 0 : mb[l];
                mi[l] = ((mb[l] & INSIDE) == INSIDE);
            }
        }
    }

    return;
}


void
mx_init(struct PD *pd, const mxArray *mxf, const mxArray *mxm)
{
    mxClassID T = mxGetClassID(mxf);
    const size_t *sz = (const size_t *)mxGetDimensions(mxf);
    const size_t szp[4] = {sz[0], sz[1], sz[2], 3};

    pd->chi  = mxCreateNumericArray(3, sz, T, mxREAL);
    pd->eta  = mxCreateNumericArray(3, sz, T, mxREAL);

    #if !THETA0
    pd->chi_ = mxCreateNumericArray(3, sz, T, mxREAL);
    pd->eta_ = mxCreateNumericArray(3, sz, T, mxREAL);
    #else
    pd->chi_ = NULL;
    pd->eta_ = NULL;
    #endif

    pd->nu = mxCreateNumericArray(3, sz, T, mxREAL);
    pd->p  = mxCreateNumericArray(4, szp, T, mxREAL);

    pd->f = mxDuplicateArray(mxf);
    pd->m = mxDuplicateArray(mxm);
    pd->mi = mxCreateNumericArray(3, sz, mxUINT8_CLASS, mxREAL);
    pd->mb = mxCreateNumericArray(3, sz, mxUINT32_CLASS, mxREAL);

    pd->chi0 = mxCreateNumericArray(3, sz, T, mxREAL);
    pd->eta0 = mxCreateNumericArray(3, sz, T, mxREAL);

    uint8_t *m  = (uint8_t *)mxGetData(pd->m);
    uint8_t *mi = (uint8_t *)mxGetData(pd->mi);
    uint32_t *mb = (uint32_t *)mxGetData(pd->mb);
    init_bitmask(mb, mi, m, sz);

    return;
}


void
mx_cleanup(struct PD *pd)
{
    if (NULL != pd->chi) {
        mxDestroyArray(pd->chi);
        pd->chi = NULL;
    }
    if (NULL != pd->eta) {
        mxDestroyArray(pd->eta);
        pd->eta = NULL;
    }
    if (NULL != pd->chi_) {
        mxDestroyArray(pd->chi_);
        pd->chi_ = NULL;
    }
    if (NULL != pd->eta_) {
        mxDestroyArray(pd->eta_);
        pd->eta_ = NULL;
    }
    if (NULL != pd->nu) {
        mxDestroyArray(pd->nu);
        pd->nu = NULL;
    }
    if (NULL != pd->p) {
        mxDestroyArray(pd->p);
        pd->p = NULL;
    }
    if (NULL != pd->f) {
        mxDestroyArray(pd->f);
        pd->f = NULL;
    }
    if (NULL != pd->m) {
        mxDestroyArray(pd->m);
        pd->m = NULL;
    }
    if (NULL != pd->mi) {
        mxDestroyArray(pd->mi);
        pd->mi = NULL;
    }
    if (NULL != pd->mb) {
        mxDestroyArray(pd->mb);
        pd->mb = NULL;
    }
    if (NULL != pd->chi0) {
        mxDestroyArray(pd->chi0);
        pd->chi0 = NULL;
    }
    if (NULL != pd->eta0) {
        mxDestroyArray(pd->eta0);
        pd->eta0 = NULL;
    }
    return;
}
