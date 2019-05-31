import numpy as np

from decomposer import *
from linearsolve import *

'''
paper:
http://faculty.baruch.cuny.edu/lwu/890/SpruillOnFiniteDifferenceMethod.doc

1. HeatEqualtion:
d_u/d_t = d^2_u/dx^2

u(x, t+dt) - u(x,t)/ dt = [ u(x+dx, t) - 2*u(x,t) + u(x-dx, t) ]/ (dx^2)

u(t, x) =  alpha*dt/(dx^2) * u(t-dt, x+dx)  +  [ 1 - 2*alpha ]*u(t-dt, x)  +  alpha*u(t-dt, x-dx) 
alpha = dt/dx^2

t - t above is tau

2. Change of variables:
    x = ln(S/K)
    tau = sigma^2*(T-t)/2
    u(x,t) = V(S,t)*e^(a*x+b*tau)/K

    a = (r-sigma^2/2.0)/sigma^2
    b = 2*r/simga^2 + a^2

BS inputs:
    S, K, sigma, r, T

Discretization inpts:
    M - number of time intervals
    alpha
    D - pricing radius


Given T and sigma in Black Scholes:
tau_final = sigma^2*T/2

M - number of time intervals
d_tau = tau_final/M

d_x = sqrt(d_tau/alpha)

x_0 = ln(S_0/K)

Reasonable choice of D is log(4)
x_left = min(x_0, 0) - D
x_right = max(x_0, 0) + D

These may not lie on the mesh, so we determine endpoints
N_left = (x_0 - x_left)/dx +1
x_left = x_0 - N_left*dx

N_right = (x_right - x_0)/dx +1
x_right = x_0 + N_right*dx

3. Boundary Conditions:

Terminal condition of call option is:
V(S,T) = max(S-K, 0)

Under change of variables we have:
u(x,0) = max(K*e^x -K, 0)*e^(a*x)/K
u(x,0) = max(e^x -1,0)*e^(a*x)
u(x_left, tau) = 0


V(S_right, t) = S_right - K*e^(-r*(T-t))
u(x_right, tau) = [K*e^(x_right) - K*e^(-r*2*tau/sigma^2)] * (e^a*x_right + b*tau)/K
u(x_right, tau) = [e^(x_right) - e^(-r*2*tau/sigma^2)]* (e^a*x_right + b*tau)

'''

class GLeft(object):

    pass

class GRight(object):

    pass

class BSCallLeft(GLeft):

    def __init__(self, s, k, vol, t, r, q):
        self.s = s
        self.k=k
        self.vol =vol
        self.t=t
        self.r=r
        self.q=q

    def get_x_left(self):
        '''
        d_x = sqrt(d_tau/alpha)
        x_0 = ln(S_0/K)
        Reasonable choice of D is log(4)
        x_left = min(x_0, 0) - D
        N_left = (x_0 - x_left)/dx +1
        x_left = x_0 - N_left*dx
        '''
        return np.log(self.s/self.k) + (self.r-self.q-self.vol*self.vol/2.0)*self.t - 3*self.vol*np.sqrt(self.t);
    
    def get(self, tau):
        '''
        Assume that S-value corresponding to x_left is sufficiently far out of money that the prob that it expires
        in the money is 0. Therefore left side boundary is 0.:
        u(x_left, tau) = 0
        '''
        return 0

class BSCallRight(GRight):

    def __init__(self, s, k, vol, t, r, q):
        self.s = s
        self.k=k
        self.vol =vol
        self.t=t
        self.r=r
        self.q=q
        self.a = (r-q)/(vol*vol)-0.5
        b= (r-q)/(vol*vol)+0.5
        self.b = b*b + 2*q/(vol*vol)
  

    def get_x_right(self):
        '''
        x_right = max(x_0, 0) + D
        N_right = (x_right - x_0)/dx +1
        x_right = x_0 + N_right*dx
        
        '''
        return np.log(self.s/self.k) + (self.r-self.q-self.vol*self.vol/2.0)*self.t + 3*self.vol*np.sqrt(self.t);


    def get(self, tau):
        '''
        Assume that S-value corresponding to x_right is sufficiently far in the money that the probability it expires
        out of the money is 0. A put would have value of zero, from put-call parity, we have that call is simply forward
        contract struck at K:
        V(S_right, t) = S_right - K*e^(-r*(T-t))
        u(x_right, tau) = [K*e^(x_right) - K*e^(-r*2*tau/sigma^2)] * (e^a*x_right + b*tau)/K
        u(x_right, tau) = [e^(x_right) - e^(-r*2*tau/sigma^2)]* (e^a*x_right + b*tau) 

        '''
        x_right = self.get_x_right()
        value = self.k*np.exp(self.a*x_right+self.b*tau)*(np.exp(x_right - 2.0*self.q*tau/(self.vol*self.vol)) - np.exp(-2.0*self.r*tau/(self.vol*self.vol)))
        return value




class FTau(object):

    pass

class BSCallTau(FTau):

    def __init__(self, s, k, vol, t, r, q):
        self.s = s
        self.k=k
        self.vol =vol
        self.t=t
        self.r=r
        self.q=q

    def get(self, x):
        '''
        Terminal condition of call option is:
        V(S,T) = max(S-K, 0)

        Under change of variables we have:
        u(x,0) = max(K*e^x -K, 0)*e^(a*x)/K
        u(x,0) = max(e^x -1,0)*e^(a*x)
        '''
        a = (self.r-self.q)/(self.vol*self.vol)-0.5
        value = self.k*np.exp(a*x)*max(np.exp(x)-1, 0)
        return value


    def get_tau_final(self):
        '''
        tau = sigma^2*(T-t)/2.0
        '''
        return self.t*self.vol*self.vol/2.0




class HeatPDESolver(object):

    def __init__(self, xleft:float, xright:float, tauFinal:float, gLeft:GLeft, gRight:GRight, f:FTau):
        self.xleft = xleft
        self.xright = xright
        self.tauFinal = tauFinal
        self.gLeft = gLeft
        self.gRight = gRight
        self.f = f


    def solve_pde(self, n:int,m:int):
        pass

