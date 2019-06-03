import numpy as np

from PDEsolver import *
from decomposer import *
from linearsolve import *

class CrankNicolson(HeatPDESolver):

    '''
    Crank Nicolson

    average b/n forward and backward

    Heat Equation by def:
    [u(x, t) - u(x,t-dt)]/ dt = [ u(x+dx, t) - 2*u(x,t) + u(x-dx, t) ]/ (dx^2) * 1/2 +
    [ u(x+dx, t-dt) - 2*u(x,t-dt) + u(x-dx, t-dt) ]/ (dx^2) * 1/2

    this becomes:
    -alpha/2*u(x-dx, t) + (1+alpha)*u(x, t) -alpha/2*u(x+dx, t) =  
    alpha/2*u(x-dx, t-dt) + (1-alpha)*u(x, t-dt) + alpha/2*u(x+dx, t-dt) =  
    ....
    u(t-dt, x) =  -alpha * u(t, x-dx)  +  [ 1 + 2*alpha ]*u(t, x)  -  alpha*u(t, x+dx) 
    u(t-dt, x+dx) =  -alpha * u(t, x)  +  [ 1 + 2*alpha ]*u(t, x+dx)  -  alpha*u(t, x+2*dx) 
    u(t-dt, x+2*dx) =  -alpha * u(t, x+dx)  +  [ 1 + 2*alpha ]*u(t, x+d2*x)  -  alpha*u(t, x+3*dx) 
    .....
    This can be solved via matrices:
    A u(t) = b(t)

    A = |1+alpha    -alpha/2        0  ...... 0 |
        |-alpha/2    1+alpha    -alpha/2 .....0 |  
        | 0         -alpha/2     1+alpha        |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |0                                    0 |

alpha=a 

 b(t) = | a/2*u(x_left, t-dt) + (1-a)*(x_left+dx,t-dt) + a/2*(x_left+2*dx,t-dt) +alpha/2*u(x_left, t)      |
        | a/2*u(x_left+dx, t-dt) + (1-a)*(x_left+2*dx,t-dt) + a/2*(x_left+3*dx,t-dt)                       |  
        |                           ................                                                       |
        |                                                                                                  |
        |                                                                                                  |
        |                                                                                                  |
        | a/2*u(x_right -3*dx, t-dt) + (1-a)*(x_right-2*dx,t-dt) + a/2*(x_right-dx,t-dt)                   |
        | a/2*u(x_right -2*dx, t-dt) + (1-a)*(x_right-dx,t-dt) + a/2*(x_right,t-dt)+ a/2*(x_right,t)       |

    the vector b(t) must be recalculated at each time step

for implementation, we have matrix B:

    B = |1-alpha    alpha/2        0  ...... 0  |
        |alpha/2    1-alpha    alpha/2 .....0   |  
        | 0         alpha/2     1-alpha         |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |0                                    0 |


    b_(t) =|  u(x_left, t)                |
           |  u(x_left + dx, t)           |
           |  u(x_left + 2*dx, t)         |
           |     ...                      |
           |  u(x_right, t)               |


    so...
    b(t) = B * b_(t)
    
    and then....
    b[0,0]  += 0.5*alpha* [ u(x_left, t-dt) + u(x_left, t) ]   //to account for first row
    b[n-2, 0] += 0.5*alpha* [ u(x_right, t) + u(x_right,t-dt) ]

    then solve...
    A u(t) = b(t)
    u(t) = A^(-1) * b(t)

    '''

    def __init__(self, xleft:float, xright:float, tauFinal:float, gLeft:GLeft, gRight:GRight, f:FTau, solver:LinearSolver):
        super().__init__(xleft, xright, tauFinal, gLeft, gRight, f)
        self.solver = solver

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

        #set up matrix A and B
        A = np.zeros((n-1, n-1))
        B = np.zeros((n-1, n-1))

        for row in range(n-1):
            A[row, row] = 1 + alpha
        for row in range(n-2):
            A[row, row+1] = -alpha/2.0
        for row in range(1, n-1):
            A[row, row-1] = -alpha/2.0

        for row in range(n-1):
            B[row, row] = 1 - alpha
        for row in range(n-2):
            B[row, row+1] = alpha/2.0
        for row in range(1, n-1):
            B[row, row-1] = alpha/2.0

        b = np.zeros(n-1)
        #u_next = np.zeros(n-1)

        for row in range(1, m+1):
            for i in range(1, n):
                b[i-1] = u[row-1, i];
            b = B @ b
            b[0] += 0.5 * alpha * (u[row, 0] + u[row-1, 0])
            b[n-2] += 0.5*alpha * (u[row, n] + u[row-1, n])

            u_next = self.solver.solve(A, b)

            for j in range(1, n):
               u[row, j] = u_next[j-1]

        return u

def solve_pde_cn(s, k, vol, t, r, alpha, time_intervals):


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

    n = N
    m = M
    #set up matrix A
    A = np.zeros((n-1, n-1))
    B = np.zeros((n-1, n-1))

    for row in range(n-1):
        A[row, row] = 1 + alpha
    for row in range(n-2):
        A[row, row+1] = -alpha/2.0
    for row in range(1, n-1):
        A[row, row-1] = -alpha/2.0

    for row in range(n-1):
        B[row, row] = 1 - alpha
    for row in range(n-2):
        B[row, row+1] = alpha/2.0
    for row in range(1, n-1):
        B[row, row-1] = alpha/2.0

    #backward
    bv = np.zeros(n-1)
    #u_next = np.zeros(n-1)

    for row in range(1, m+1):
        for i in range(1, n):
            bv[i-1] = u[i, row-1]
        bv = B @ bv
        bv[0] +=  0.5*alpha*(u[0,row] + u[0, row-1])
        bv[n-2] +=  0.5* alpha*(u[n,row] + u[n, row-1])

        u_next = LuNoPivSolve.solve(A, bv)

        for j in range(1, n):
            u[j, row] = u_next[j-1]

    v = k * u[N_left,M]*np.exp(-a*x_0 - b*tau_final)
    return v

