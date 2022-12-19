import numpy as np
import numba as nb

 # consav
from consav import linear_interp # for linear interpolation

import funcs

@nb.njit(parallel=True)
def lifecycle(sim,sol,par):
    """ simulate full life-cycle """

    # unpack (to help numba optimize)
    id = sim.id
    time = sim.time
    p = sim.p
    m = sim.m
    c = sim.c
    a = sim.a
    y_pre = sim.y_pre
    y = sim.y
    MPC = sim.MPC
    MPCP_perm = sim.MPCP_perm
    MPCP_pers = sim.MPCP_pers
    MPCP_scale = sim.MPCP_scale
    
    for t in range(par.simT):
        for i in nb.prange(par.simN):

            id[t,i] = i
            time[t,i] = t
            
            # a. beginning of period states
            if t == 0:
                p_lag = 1.0
                a_lag = 0.0
            else:
                p_lag = p[t-1,i]
                a_lag = a[t-1,i]
            
            p[t,i] = funcs.p(t,p_lag,sim.psi[t,i],par)
            y[t,i],y_pre[t,i] = funcs.y(t,sim.alpha[i],p[t,i],sim.xi[t,i],False,par)
            m[t,i] = par.R*a_lag + y[t,i]

            # b. choices
            c[t,i] = linear_interp.interp_4d(par.grid_alpha,par.grid_beta,par.grid_p,par.grid_m,sol.c[t,0],sim.alpha[i],sim.beta[i],p[t,i],m[t,i])
            a[t,i] = m[t,i]-c[t,i]

            # c. MPC and MPCP

            # MPCP
            c_MPC = linear_interp.interp_4d(par.grid_alpha,par.grid_beta,par.grid_p,par.grid_m,sol.c[t,0],sim.alpha[i],sim.beta[i],p[t,i],m[t,i]+par.delta)
            
            MPC[t,i] = (c_MPC-c[t,i])/par.delta
            
            # MPCP_perm
            m_MPCP_perm = m[t,i] + par.delta
            c_MPCP_perm = linear_interp.interp_4d(par.grid_alpha,par.grid_beta,par.grid_p,par.grid_m,sol.c[t,1],sim.alpha[i],sim.beta[i],p[t,i],m_MPCP_perm)
            
            MPCP_perm[t,i] = (c_MPCP_perm-c[t,i])/par.delta

            # MPCP_pers
            p_pers = p[t,i] + par.delta
            y_MPCP_pers,_ = funcs.y(t,sim.alpha[i],p_pers,sim.xi[t,i],False,par)
            m_MPCP_pers = par.R*a_lag + y_MPCP_pers

            c_MPCP_pers = linear_interp.interp_4d(par.grid_alpha,par.grid_beta,par.grid_p,par.grid_m,sol.c[t,0],sim.alpha[i],sim.beta[i],p_pers,m_MPCP_pers)
            MPCP_pers[t,i] = (c_MPCP_pers-c[t,i])/(y_MPCP_pers-y[t,i])

            # MPCP_scale
            alpha_MPCP_scale = sim.alpha[i]*(1+par.delta)
            a_lag_MPCP_scale = a_lag*(1+par.delta)
            y_MPCP_scale,_ = funcs.y(t,alpha_MPCP_scale,p[t,i],sim.xi[t,i],False,par)
            m_MPCP_scale = par.R*a_lag_MPCP_scale + y_MPCP_scale

            c_MPCP_scale = linear_interp.interp_4d(par.grid_alpha,par.grid_beta,par.grid_p,par.grid_m,sol.c[t,0],alpha_MPCP_scale,sim.beta[i],p[t,i],m_MPCP_scale)

            MPCP_scale[t,i] = (np.log(c_MPCP_scale)-np.log(c[t,i]))/(np.log(alpha_MPCP_scale)-np.log(sim.alpha[i]))