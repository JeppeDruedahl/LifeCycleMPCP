# -*- coding: utf-8 -*-
"""LifeCycleModel

"""

##############
# 1. imports #
##############

import time
import numpy as np

# consav package
from consav import ModelClass, jit
from consav.grids import nonlinspace
from consav.quadrature import create_PT_shocks
from consav.misc import elapsed

# local modules
import funcs
import post_decision
import egm
import simulate

############
# 2. model #
############

class LifeCycleModelModelClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
    def setup(self):
        """ set baseline parameters """   

        par = self.par

        # a. horizon
        par.T = 55
        par.Tr = 40
        
        # c. preferences
        par.mu_beta = 0.96
        par.sigma_beta = 0.01
        
        par.sigma0 = 2.0
        par.omega = 1.0 

        par.nu = 15.0
        par.aubar = 0.0

        par.iota = 0.0

        # d. returns 
        par.R = 1.03

        # profile
        par.rho = 0.95
        par.G = 1.02
        par.phi = 0.60

        # types
        par.mu_alpha = 1.0
        par.sigma_alpha = 0.2

        # tax
        par.tau = 0.0
        par.kappa = 1.5

        # risk
        par.sigma_psi = 0.10
        par.Npsi = 4
        par.sigma_xi = 0.1
        par.Nxi = 1
        par.pi = 0.05
        par.mu = 0.50
        
        # e. grids (number of points)
        par.Nalpha = 20
        par.Nbeta = 10

        par.Np = 50
        par.p_min = 0.1
        par.p_max = 10

        par.Na = 200
        par.a_max = 50

        par.Nm = 100

        # f. misc
        par.delta = 1e-4
        par.tol = 1e-8
        par.do_print = True

        # g. simulation
        par.simT = par.T
        par.simN = 500_000
        par.sim_seed = 1998
        
    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        self.create_grids()
        self.solve_prep()
        self.simulate_prep()

    def create_grids(self):
        """ construct grids for states and shocks """

        par = self.par

        # a. states (unequally spaced vectors of length Nm)
        diff_alpha = par.sigma_alpha+par.iota*par.sigma_beta
        par.grid_alpha = nonlinspace(par.mu_alpha-diff_alpha,par.mu_alpha+diff_alpha,par.Nalpha,1.0)
        par.grid_beta = nonlinspace(par.mu_beta-par.sigma_beta,par.mu_beta+par.sigma_beta,par.Nbeta,1.0)
        par.grid_m = nonlinspace(1e-6,20,par.Nm,1.1)
        par.grid_p = nonlinspace(1e-4,10,par.Np,1.1)
        
        # b. post-decision states (unequally spaced vector of length Na)
        par.grid_a = nonlinspace(1e-6,20,par.Na,1.1)
        
        # c. shocks (qudrature nodes and weights using GaussHermite)
        shocks = create_PT_shocks(
            par.sigma_psi,par.Npsi,par.sigma_xi,par.Nxi,
            par.pi,par.mu)
        par.psi,par.psi_w,par.xi,par.xi_w,par.Nshocks = shocks

        # d. set seed
        np.random.seed(self.par.sim_seed)

    #########
    # solve #
    #########

    def solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol

        sol.c = np.nan*np.ones((par.T,2,par.Nalpha,par.Nbeta,par.Np,par.Nm))        
        sol.q = np.nan*np.zeros((2,par.Nalpha,par.Nbeta,par.Np,par.Na))

    def solve(self):
        """ solve the model using solmethod """

        with jit(self) as model: # can now call jitted functions

            par = model.par
            sol = model.sol

            # backwards induction
            for t in reversed(range(par.T)):
                
                t0 = time.time()
                
                # a. compute post-decision functions
                t0_w = time.time()

                if t == par.T-1:
                    post_decision.compute_q_last(sol,par)
                else:
                    post_decision.compute_q(t,sol,par)

                t1_w = time.time()

                # b. solve bellman equation
                egm.solve_bellman(t,sol,par)                    

                # c. print
                if par.do_print:
                    msg = f' t = {t} solved in {elapsed(t0)}'
                    if t < par.T-1:
                        msg += f' (post-decision: {elapsed(t0_w,t1_w)})'                
                    print(msg)

    ############
    # simulate #
    ############

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim
        
        sim.id = np.zeros((par.simT,par.simN))
        sim.time = np.zeros((par.simT,par.simN))

        sim.alpha = np.zeros(par.simN)
        sim.beta = np.zeros(par.simN)

        sim.p = np.zeros((par.simT,par.simN))
        sim.m = np.zeros((par.simT,par.simN))
        sim.c = np.zeros((par.simT,par.simN))
        sim.a = np.zeros((par.simT,par.simN))
        sim.y_pre = np.zeros((par.simT,par.simN))
        sim.y = np.zeros((par.simT,par.simN))

        sim.psi = np.ones((par.simT,par.simN))
        sim.xi = np.ones((par.simT,par.simN))

        sim.MPC = np.ones((par.simT,par.simN))
        sim.MPCP_perm = np.ones((par.simT,par.simN))
        sim.MPCP_pers = np.ones((par.simT,par.simN))
        sim.MPCP_scale = np.ones((par.simT,par.simN))

    def simulate(self):
        """ simulate model """

        with jit(self) as model: # can now call jitted functions 

            par = model.par
            sol = model.sol
            sim = model.sim
            
            t0 = time.time()

            assert par.simN%par.Nalpha*par.Nbeta == 0
            
            # a. allocate memory and draw random numbers
            simN_shock = par.simN//(par.Nalpha*par.Nbeta)
            psi = np.exp(-0.5*par.sigma_psi**2 + par.sigma_psi*np.random.normal(0,1,size=(par.T,simN_shock)))
            xi = (np.exp(-0.5*par.sigma_xi**2 + par.sigma_xi*np.random.normal(0,1,size=(par.T,simN_shock))) - par.mu*par.pi)/(1-par.pi)
            I = np.random.uniform(0,1,size=(par.T,simN_shock)) < par.pi
            xi[I] = par.mu
            
            i = 0
            for ialpha in range(par.Nalpha):
                for ibeta in range(par.Nbeta):

                    sim.alpha[i*simN_shock:(i+1)*simN_shock] = par.grid_alpha[ialpha] + par.iota*(par.grid_beta[ibeta]-par.mu_beta)
                    sim.beta[i*simN_shock:(i+1)*simN_shock] = par.grid_beta[ibeta]

                    sim.psi[:,i*simN_shock:(i+1)*simN_shock] = psi
                    sim.xi[:,i*simN_shock:(i+1)*simN_shock] = xi

                    i += 1

            # retirement
            sim.psi[par.Tr:,:] = 1.0
            sim.xi[par.Tr:,:] = 1.0

            # b. simulate
            simulate.lifecycle(sim,sol,par)

        if par.do_print:
            print(f'model simulated in {elapsed(t0)}')