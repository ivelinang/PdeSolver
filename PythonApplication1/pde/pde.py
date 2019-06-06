import numpy as np

from linearsolve import *


def solve_pde_fe_call(s, k, vol, t, r, spot_intervals, time_intervals):


    #time intervals
    tau_final = t
    d_tau = tau_final/time_intervals

    S_min = 0
    S_max = s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
    d_x = (S_max - S_min)/spot_intervals

    M = spot_intervals
    N = time_intervals
    u = np.zeros((M+1, N+1)) # 0->M , 0->N

    #initial boundary conditions
    for i in range(N+1):        
        tau_i = i*d_tau
        u[0,i] = 0 #left boundary
        u[M, i] = S_max - k*np.exp(-r*tau_i)   #right boundary S - Ke^(-r(T-t))

    #side boundary conditions
    for j in range(M+1):
        x_i = S_min +j*d_x
        u[j,N] = max(x_i - k, 0)


    alpha = lambda j: 0.5*d_tau*(r*j-vol*vol*j*j)
    beta = lambda j: 1.0+d_tau*(vol*vol*j*j+r)
    gamma = lambda j: -0.5*d_tau*(r*j+vol*vol*j*j)

    #now we need
    #A * Pi  = Pi+1 - Ci
    #A is tridag matrix
    '''
     A = |beta1    gamma1        0  .   ..... 0 |
        |alpha2     beta2    gamma2      .....0 |  
        | 0         alpha3                      |
        |                                       |
        |                                       |
        |                                       |
        |                                       |
        |0                                    0 |
    '''

    #set up matrix A
    A = np.zeros((M-1, M-1))
    for row in range(M-1):
        A[row, row] = beta(row+1)
    for row in range(M-2):
        A[row, row+1] = gamma(row+1)
    for row in range(1, M-1):
        A[row, row-1] = alpha(row+1)

    C = np.zeros(M-1)
    

    for i in range(N-1, -1, -1):
        C[0] = alpha(1) * u[0,  i]
        C[-1] = gamma(M) * u[-1, i]
        P_i1 = u[1:M, i+1]
        P_C = P_i1 - C
        u_next = LuNoPivSolve.solve(A, P_C)
        u[1:M, i] = u_next




    i_low = int((s-S_min)/d_x)
    i_high = i_low + 1

    v_low = u[i_low,0]
    v_high = u[i_high, 0]

    s_low = S_min+ i_low*d_x
    s_high = S_min+ i_high*d_x

    price = ((s_high - s) * v_low + (s - s_low) * v_high) / (s_high - s_low);
    return price
    
   
