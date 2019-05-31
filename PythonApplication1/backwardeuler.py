import numpy as np

from PDEsolver import *
from decomposer import *
from linearsolve import *

class BackwardEuler(HeatPDESolver):

    '''
    Backward Euler

    Heat Equation by def:
    u(x, t) - u(x,t-dt)/ dt = k^2*[ u(x+dx, t) - 2*u(x,t) + u(x-dx, t) ]/ (dx^2)

    this becomes:
    u(t-dt, x) =  -k^2*dt/(dx^2) * u(t, x-dx)  +  [ 1 + 2*k^2*dt/(dx^2) ]*u(t, x)  -  k^2*dt/(dx^2)*u(t, x+dx) 

    This can be written in matrix form:
    A u(t) = u(t-dt) + c(t)

    factoring the matrix A is more time-efficient than inverting it, 
    and solving the system of equations using the factorization is no more expensive 
    than a matrix multiplication involving the inverse. 
    The method described here uses LU decomposition to accomplish the iteration.

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

        #set up matrix A
        A = np.zeros((n-1, n-1))
        for row in range(n-1):
            A[row, row] = 1 + 2*alpha
        for row in range(n-2):
            A[row, row+1] = -alpha
        for row in range(1, n-1):
            A[row, row-1] = -alpha

        #backward
        b = np.zeros(n-1)
        u_next = np.zeros(n-1)

        for row in range(1, m+1):
            for i in range(1, n):
                b[i-1] = u[row-1, i]
            b[0] = b[0] + alpha*u[row,0]
            b[n-2] = b[n-2] + alpha*u[row,n]

            u_next = self.solver.solve(A, b)

            for j in range(1, n):
                u[row, j] = u_next[j-1]

        return u

def solve_pde_be(s, k, vol, t, r, alpha, time_intervals):


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
    for row in range(n-1):
        A[row, row] = 1 + 2*alpha
    for row in range(n-2):
        A[row, row+1] = -alpha
    for row in range(1, n-1):
        A[row, row-1] = -alpha

    #backward
    bv = np.zeros(n-1)
    #u_next = np.zeros(n-1)

    for row in range(1, m+1):
        for i in range(1, n):
            bv[i-1] = u[i, row-1]
        bv[0] = bv[0] + alpha*u[0,row]
        bv[n-2] = bv[n-2] + alpha*u[n,row]

        u_next = LuNoPivSolve.solve(A, bv)

        for j in range(1, n):
            u[j, row] = u_next[j-1]

    v = k * u[N_left,M]*np.exp(-a*x_0 - b*tau_final)
    return v



