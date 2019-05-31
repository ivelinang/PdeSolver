import numpy as np

from PDEsolver import *
from decomposer import *
from linearsolve import *

class ForwardEuler(HeatPDESolver):

    '''
    Forward Euler

    Heat Equation by def:
    u(x, t+dt) - u(x,t)/ dt = k^2*[ u(x+dx, t) - 2*u(x,t) + u(x-dx, t) ]/ (dx^2)

    ..then

    u(x, t+dt) = u(x,t) + k^2*dt/(dx^2) * [ u(x+dx, t) - 2*u(x,t) + u(x-dx, t) ]

    ...or

    u(x, t) = u(x,t-dt) + k^2*dt/(dx^2) * [ u(x+dx, t-dt) - 2*u(x,t-dt) + u(x-dx, t-dt) ]

    .... equivalent

    u(t, x) = u(t-dt, x) + k^2*dt/(dx^2) * [ u(t-dt, x+dx) - 2*u(t-dt, x) + u(t-dt, x-dx) ]


    ... equivalent

    u(t, x) =  k^2*dt/(dx^2) * u(t-dt, x+dx)  +  [ 1 - 2*k^2*dt/(dx^2) ]*u(t-dt, x)  +  k^2*dt/(dx^2)*u(t-dt, x-dx) 
    
    x = col (c)
    t = row (r)
    k^2 *dt/(dx^2) = alpha

    .. below formula is the same

    u[row, col] = alpha * u[row-1, col+1] + (1-2*alpha)*u[row-1, col] + alpha*u[row-1, col-1]


    --The matrix u is below

    right is N - stock price
    down is M - time

         0 1 2 3 4 5 6 .......                  N+1
         _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
       0 |Ftau(xleft + i*delta_x).............. Ftau(xright)
       1 |gLeft(j*delta_t)  ..........          gRight(j*delta_t)
       2 | .......                                .....
       3 |
       4 |
         |
         | ........                                ........
   M+1   |gLeft(j*delta_t)  ..........           gRight(j*delta_t)



            0 dt 2*dt 3*dt ....... t (today) ................................T
         _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
       0 |                          |                                          | 
      dS |                          |                                          |
    2*dS |                          |                                          | 
    3*dS |                          |                                          | terminal boundary (as t-> T maturity) -> V = max(S(T) - K, 0)
       . |                          |                                          | 
 S today |--------------------------V(i,j) - this is what we need              |
         |                                                                     |
       . |                                                                     |
   Smax  |---------------------------------------------------------------------
                    uppper boundary for S

    The first column:
    for i in range(0, Smax):
        U[0, i] = max(S-K,0) // this is a bit confusing but the terminal condition becomes initial conditions under change of variables!!!

    The first row:
    for i in range(0, T):
        U[i, 0] = 0    // call has no probability to go ITM, regardless of time to expiry as S is very low, V=0

    The last row:
    for i in range(0, T):
        U[Smax, i] = Smax - K*e^(r(T-t))  //call is defintely ITM ,so just discount the payoff of the option

    '''

    def __init__(self, xleft:float, xright:float, tauFinal:float, gLeft:GLeft, gRight:GRight, f:FTau):
        super().__init__(xleft, xright, tauFinal, gLeft, gRight, f)

    def solve_pde(self, n, m):

        delta_t = self.tauFinal/float(m)
        delta_x = (self.xright - self.xleft)/float(n)
        alpha = delta_t / (delta_x * delta_x)

        #set up boundary conditions
        u = np.zeros((m+1, n+1))

        for i in range(n+1):
            u[0,i] = self.f.get(self.xleft + i*delta_x)

        for j in range(1, m+1):
            u[j,0] = self.gLeft.get(j*delta_t)
            u[j,n] = self.gRight.get(j*delta_t)

        #use forward euler to compute nodes
        for row in range(1, m+1):
            for col in range(1, n):
                u[row, col] = alpha * u[row-1, col+1] + (1-2*alpha)*u[row-1, col] + alpha*u[row-1, col-1]

        return u



        
def solve_pde(s, k, vol, t, r, alpha, time_intervals):

    tau_final = t*vol*vol/2.0
    d_tau = tau_final/time_intervals
    d_x = np.sqrt(d_tau/alpha)
    x_0 = np.log(s/k)
    x_right_initial = max(x_0, 0) + np.log(4)
    N_right = int((x_right_initial - x_0)/d_x) +1
    x_right = x_0 + N_right*d_x

    x_left_initial = min(x_0, 0) - np.log(4)
    N_left = int((x_0 - x_left_initial)/d_x) +1
    x_left = x_0 - N_left*d_x

    a = (r - vol*vol/2.0)/(vol*vol)
    b = 2*r/(vol*vol) + a*a


    N = N_left + N_right
    M = time_intervals
    #grid mesh
    u = np.zeros((N+1, M+1))

    #initial boundary conditions
    for i in range(N+1):
        x_i = x_left + i*d_x
        u[i,0] = max(np.exp(x_i)-1.0, 0)*np.exp(a*x_i)

    #side boundary conditions
    for j in range(1, M+1):
        tau_i = j*d_tau
        u[0,j] = 0
        u[N,j] = np.exp(a*x_right + b*tau_i) * (np.exp(x_right) - np.exp(-r*2*tau_i/(vol*vol)))

    #iterate with loop
    #use forward euler to compute nodes
    for row in range(1, M+1):
        for col in range(1, N):
            u[col, row] = alpha * u[col+1, row-1] + (1-2*alpha)*u[col, row-1] + alpha*u[col-1, row-1]


    v = k * u[N_left,M]*np.exp(-a*x_0 - b*tau_final)
    return v


