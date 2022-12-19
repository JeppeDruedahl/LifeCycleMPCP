import numpy as np
import numba as nb


@nb.njit
def marg_func(t,c,par):

    sigma = par.sigma0*par.omega**np.fmin(t,par.Tr)
    return c**(-sigma)

@nb.njit
def inv_marg_func(t,q,par):

    sigma = par.sigma0*par.omega**np.fmin(t,par.Tr)
    return q**(-1/sigma)

@nb.njit
def p(t,p_lag,psi,par):

    if t <= par.Tr:

        p = p_lag**par.rho*psi
        p = np.fmin(p,par.p_max)
        p = np.fmax(p,par.p_min)

    else:

        p = p_lag

    return p

@nb.njit
def y(t,alpha,p,xi,add,par):

    # a. pre-tax
    Gt = par.G**np.fmin(t,par.Tr)
    y_pre = alpha*Gt*p*xi

    # b. retirement
    if t >= par.Tr:
        y_pre *= par.phi

    # c. y - post-tax
    y_post = y_pre - par.tau*np.fmax(y_pre-par.kappa,0.0)

    # d. MPCP
    if add:
        y_post += par.delta

    return y_post,y_pre