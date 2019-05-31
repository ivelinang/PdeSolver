import numpy as np

from PDEsolver import *

class BsPdeSolver(object):

    def __init__(self, s, k, vol, t, r, q, solver):
        self.s = s
        self.k=k
        self.vol =vol
        self.t=t
        self.r=r
        self.q=q
        self.solver = solver
        self.a = (r-q)/(vol*vol)-0.5
        b= (r-q)/(vol*vol)+0.5
        self.b = b*b + 2*q/(vol*vol)
            
    def solve_pde(self, alpha, m:int):
        N = self.compute_n(alpha, m)

        soln = self.solver.solve_pde(N, m)
        return soln
 
    def compute_price_alpha(self, alpha, m):
        soln = self.solve_pde(alpha, m)

        N = self.compute_n(alpha, m)
        return self.compute_price_(N, m, soln)    

    def compute_price_(self, n:int, m:int, soln):
        #soln = self.solve_pde(alpha, m)

        N = n # self.compute_n(alpha, m)

        #find i s.t. x_i <= x_compute <= x_i+1
        x_compute = np.log(self.s/self.k)
        x_left = np.log(self.s/self.k) + (self.r - self.q - self.vol*self.vol/2.0)*self.t - 3*self.vol*np.sqrt(self.t)

        delta_x = (6.0* self.vol * np.sqrt(self.t)) / float(N)

        i = int((x_compute - x_left)/delta_x)

        tau_final = self.t*self.vol*self.vol/2.0

        if i+1 > N:
            return -1

        x_1 = x_left + i*delta_x
        x_2 = x_left + (i+1)*delta_x

        v_1 = np.exp(-self.a*x_1 - self.b*tau_final) * soln[m, i]
        v_2 = np.exp(-self.a*x_2 - self.b*tau_final) * soln[m, i+1];

        s_1 = self.k * np.exp(x_1);
        s_2 = self.k * np.exp(x_2);    

        #interpolate b/n two points
        price = ((s_2 - self.s) * v_1 + (self.s - s_1) * v_2) / (s_2 - s_1);
        return price


    def compute_n(self, alpha, m:int):
        tau_final = self.t * self.vol * self.vol / 2.0
        delta_tau = tau_final/float(m)

        #x_right - x_left = 6* vol * sqrt(t) ,   six s.d. ?
        rl = 6*self.vol*np.sqrt(self.t)

        float_n = rl / np.sqrt(delta_tau/alpha)

        n = int(float_n)

        return n

