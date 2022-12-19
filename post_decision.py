import numpy as np
import numba as nb

# consav
from consav import linear_interp # for linear interpolation

# local modules
import funcs

@nb.njit(parallel=True)
def compute_q_last(sol,par):
    """ compute the post-decision function q """

    # unpack (helps numba optimize)
    q = sol.q

    # loop over outermost post-decision state
    for ialpha in nb.prange(par.Nalpha):
        for ibeta in nb.prange(par.Nbeta):
            for ip in nb.prange(par.Np):
                for iadd in range(2):
                    
                    beta = par.grid_beta[ibeta]
                    sigma = par.sigma0*par.omega**par.Tr

                    q[iadd,ialpha,ibeta,ip,:] = beta*par.nu*(par.grid_a+par.aubar)**(-sigma)

@nb.njit(parallel=True)
def compute_q(t,sol,par):
    """ compute the post-decision function q """

    # unpack (helps numba optimize)
    q = sol.q

    # loop over outermost post-decision state
    for ialpha in nb.prange(par.Nalpha):
        for ibeta in nb.prange(par.Nbeta):
            for ip in nb.prange(par.Np):
                for iadd in range(2):

                    # a. allocate containers and initialize at zero
                    q[iadd,ialpha,ibeta,ip,:] = 0
                    c_plus = np.empty(par.Na)

                    # b. states
                    alpha = par.grid_alpha[ialpha]
                    beta = par.grid_beta[ibeta]
                    p = par.grid_p[ip]

                    # c. loop over shocks and then end-of-period assets
                    working = t+1 < par.Tr
                    Nshocks = par.Nshocks if working else 1
                    for ishock in range(Nshocks):
                        
                        # i. shocks
                        if working:
                            psi_plus = par.psi[ishock]
                            psi_plus_w = par.psi_w[ishock]
                            xi_plus = par.xi[ishock]
                            xi_plus_w = par.xi_w[ishock]
                        else:
                            psi_plus = 1.0
                            psi_plus_w = 1.0
                            xi_plus = 1.0
                            xi_plus_w = 1.0

                        # ii. next-period income
                        p_plus = funcs.p(t+1,p,psi_plus,par)
                        y_plus,_y_pre_plus = funcs.y(t+1,alpha,p,xi_plus,iadd,par)
                        
                        # iii. prepare interpolation in p direction
                        prep = linear_interp.interp_2d_prep(par.grid_p,p_plus,par.Na)

                        # iv. weight
                        weight = psi_plus_w*xi_plus_w

                        # v. next-period cash-on-hand and interpolate
                        m_plus = par.R*par.grid_a + y_plus
                        linear_interp.interp_2d_only_last_vec_mon(prep,par.grid_p,par.grid_m,sol.c[t+1,iadd,ialpha,ibeta],p_plus,m_plus,c_plus)

                        # vi. accumulate all
                        q[iadd,ialpha,ibeta,ip,:] += weight*par.R*beta*funcs.marg_func(t+1,c_plus,par)