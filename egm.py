import numpy as np
import numba as nb

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import funcs

@nb.njit(parallel=True)
def solve_bellman(t,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    c = sol.c[t]

    for ialpha in nb.prange(par.Nalpha):
        for ibeta in nb.prange(par.Nbeta):
            for ip in nb.prange(par.Np):
                for iadd in range(2):
                
                    # a. temporary container (local to each thread)
                    m_temp = np.zeros(par.Na+1) # m_temp[0] = 0
                    c_temp = np.zeros(par.Na+1) # c_temp[0] = 0

                    # b. invert Euler equation
                    for ia in range(par.Na):
                        c_temp[ia+1] = funcs.inv_marg_func(t,sol.q[iadd,ialpha,ibeta,ip,ia],par)
                        m_temp[ia+1] = par.grid_a[ia] + c_temp[ia+1]
                    
                    # b. re-interpolate consumption to common grid
                    linear_interp.interp_1d_vec_mon_noprep(m_temp,c_temp,par.grid_m,c[iadd,ialpha,ibeta,ip,:])