import numpy as np

from linearsolve import *

from scipy.interpolate import interp1d


def solve_pde_fe_generic(s, k, vol, t, r, spot_intervals, time_intervals, call=True, S_max=None):


    #time intervals
    tau_final = t
    d_tau = tau_final/time_intervals

    S_min = 0
    if S_max is None:
        log_mean = np.log(s)+(r-vol*vol*0.5)*t
        log_sd = vol*np.sqrt(t)
        log_big_value = log_mean + 6*log_sd #6 standard deviations away from mean
        big_value = np.exp(log_big_value)
        S_max =big_value #s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
    # IF YOU WANT TO MATCH THE PAPER
    # https://www.scribd.com/document/317457984/Solution-to-Black-Scholes-P-D-E-via-Finite-Difference-Methods
    #S_max = 250 # make S-max = 250 to match paper
    d_x = (S_max - S_min)/spot_intervals

    M = spot_intervals
    N = time_intervals
    u = np.zeros((M+1, N+1)) # 0->M , 0->N

    if call:
        left_boundary =lambda S,T,K,R: 0
        right_boundary = lambda S,T,K,R: S - K*np.exp(-R*T)
        side_boundary = lambda S,K : max(S-K, 0)
    else:
        left_boundary =lambda S,T,K,R,: K*np.exp(-R*T)
        right_boundary = lambda S,T,K,R: 0
        side_boundary = lambda S,K : max(K-S, 0)

    #initial boundary conditions
    for i in range(N+1):        
        tau_i = i*d_tau
        u[0,i] = left_boundary(S_min, tau_i, k, r)#k*np.exp(-r*tau_i) #left boundary
        u[M, i] = right_boundary(S_max, tau_i, k, r) #0 #S_max - k*np.exp(-r*tau_i)   #right boundary S - Ke^(-r(T-t))

    #side boundary conditions
    for j in range(M+1):
        x_i = S_min +j*d_x
        u[j,N] = side_boundary(x_i, k)#max(k - x_i, 0)        #max(x_i - k, 0)


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
        #u_next = LuNoPivSolve.solve(A, P_C)
        u_next = np.linalg.solve(A, P_C)
        u[1:M, i] = u_next




    i_low = int((s-S_min)/d_x)
    i_high = i_low + 1

    v_low = u[i_low,0]
    v_high = u[i_high, 0]

    s_low = S_min+ i_low*d_x
    s_high = S_min+ i_high*d_x

    price = ((s_high - s) * v_low + (s - s_low) * v_high) / (s_high - s_low);
    return price
    
   
def solve_pde_be_generic(s, k, vol, t, r, spot_intervals, time_intervals, call=True, S_max=None):


    #time intervals
    tau_final = t
    d_tau = tau_final/time_intervals

    S_min = 0
    if S_max is None:
        log_mean = np.log(s)+(r-vol*vol*0.5)*t
        log_sd = vol*np.sqrt(t)
        log_big_value = log_mean + 6*log_sd #6 standard deviations away from mean
        big_value = np.exp(log_big_value)
        S_max =big_value #s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
    #S_max = 250 #s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
    d_x = (S_max - S_min)/spot_intervals

    M = spot_intervals
    N = time_intervals
    u = np.zeros((M+1, N+1)) # 0->M , 0->N

    if call:
        left_boundary =lambda S,T,K,R: 0
        right_boundary = lambda S,T,K,R: S - K*np.exp(-R*T)
        side_boundary = lambda S,K : max(S-K, 0)
    else:
        left_boundary =lambda S,T,K,R,: K*np.exp(-R*T)
        right_boundary = lambda S,T,K,R: 0
        side_boundary = lambda S,K : max(K-S, 0)

    #initial boundary conditions
    for i in range(N+1):        
        tau_i = i*d_tau
        u[0,i] = left_boundary(S_min, tau_i, k, r)#k*np.exp(-r*tau_i) #left boundary
        u[M, i] = right_boundary(S_max, tau_i, k, r) #0 #S_max - k*np.exp(-r*tau_i)   #right boundary S - Ke^(-r(T-t))

    #side boundary conditions
    for j in range(M+1):
        x_i = S_min +j*d_x
        u[j,N] = side_boundary(x_i, k)#max(k - x_i, 0)        #max(x_i - k, 0)


    alpha = lambda j: -0.5*d_tau*(r*j-vol*vol*j*j)
    beta = lambda j: 1.0-d_tau*(vol*vol*j*j+r)
    gamma = lambda j: 0.5*d_tau*(r*j+vol*vol*j*j)

    #use backward euler to compute nodes
    for i in range(N-1, -1, -1):
        for j in range(1, M):
            u[j, i] = alpha(j) * u[j-1, i+1] + beta(j)*u[j,i+1] + gamma(j)*u[j+1, i+1]

    i_low = int((s-S_min)/d_x)
    i_high = i_low + 1

    v_low = u[i_low,0]
    v_high = u[i_high, 0]

    s_low = S_min+ i_low*d_x
    s_high = S_min+ i_high*d_x

    price = ((s_high - s) * v_low + (s - s_low) * v_high) / (s_high - s_low);
    return price   

def solve_pde_cn_generic(s, k, vol, t, r, spot_intervals, time_intervals, call=True, S_max=None):


    #time intervals
    tau_final = t
    d_tau = tau_final/time_intervals

    S_min = 0
    if S_max is None:
        log_mean = np.log(s)+(r-vol*vol*0.5)*t
        log_sd = vol*np.sqrt(t)
        log_big_value = log_mean + 6*log_sd #6 standard deviations away from mean
        big_value = np.exp(log_big_value)
        S_max =big_value #s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
    d_x = (S_max - S_min)/spot_intervals
    # IF YOU WANT TO MATCH THE PAPER
    # https://www.scribd.com/document/317457984/Solution-to-Black-Scholes-P-D-E-via-Finite-Difference-Methods
    #S_max = 250 # make S-max = 250 to match paper

    M = spot_intervals
    N = time_intervals
    u = np.zeros((M+1, N+1)) # 0->M , 0->N

    if call:
        left_boundary =lambda S,T,K,R: 0
        right_boundary = lambda S,T,K,R: S - K*np.exp(-R*T)
        side_boundary = lambda S,K : max(S-K, 0)
    else:
        left_boundary =lambda S,T,K,R,: K*np.exp(-R*T)
        right_boundary = lambda S,T,K,R: 0
        side_boundary = lambda S,K : max(K-S, 0)

    #initial boundary conditions
    for i in range(N+1):        
        tau_i = i*d_tau
        u[0,i] = left_boundary(S_min, tau_i, k, r)#k*np.exp(-r*tau_i) #left boundary
        u[M, i] = right_boundary(S_max, tau_i, k, r) #0 #S_max - k*np.exp(-r*tau_i)   #right boundary S - Ke^(-r(T-t))

    #side boundary conditions
    for j in range(M+1):
        x_i = S_min +j*d_x
        u[j,N] = side_boundary(x_i, k)#max(k - x_i, 0)        #max(x_i - k, 0)


    alpha = lambda j: -0.25*(r*j+vol*vol*j*j)
    ceta = lambda j: 0.25*(r*j-vol*vol*j*j)
    beta = lambda j: 1.0/d_tau+0.5*r+0.5*vol*vol*j*j
    deta = lambda j: 1.0/d_tau-0.5*r-0.5*vol*vol*j*j

    #now we need
    #A * Pi  = Pi+1 - Ci
    #A is tridag matrix
    '''
     A = |beta1    alpha1        0  .   ..... 0 |
        |ceta2     beta2    alpha2      .....0 |  
        | 0         ceta3                      |
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
        A[row, row+1] = alpha(row+1)
    for row in range(1, M-1):
        A[row, row-1] = ceta(row+1)

    #set up matrix B
    B = np.zeros((M-1, M-1))
    for row in range(M-1):
        B[row, row] = deta(row+1)
    for row in range(M-2):
        B[row, row+1] = -alpha(row+1)
    for row in range(1, M-1):
        B[row, row-1] = -ceta(row+1)

    #C = np.zeros(M-1)   

    for i in range(N-1, -1, -1):

        b = u[1:M, i+1] # get previous time vector
        b = B @ b # multiply by B matrix
        b[0] = -ceta(1)*(u[0,i+1]+u[0,i])
        b[M-2] = -alpha(M-1)*(u[M,i+1]+u[M,i])        
        u_next = np.linalg.solve(A, b)
        u[1:M, i] = u_next




    i_low = int((s-S_min)/d_x)
    i_high = i_low + 1

    v_low = u[i_low,0]
    v_high = u[i_high, 0]

    s_low = S_min+ i_low*d_x
    s_high = S_min+ i_high*d_x

    price = ((s_high - s) * v_low + (s - s_low) * v_high) / (s_high - s_low);

    #or
    S = np.zeros(M+1)
    for j in range(M+1):
        S[j] = S_min +j*d_x
    f =interp1d(S, u[:, 0], kind='linear')
    return f(s)
    #return price


def solve_pde_fe_better(s, k, vol, t, r, spot_intervals, time_intervals, call=True, S_max=None):


    #time intervals
    tau_final = t
    d_tau = tau_final/time_intervals

    S_min = 0
    if S_max is None:
        log_mean = np.log(s)+(r-vol*vol*0.5)*t
        log_sd = vol*np.sqrt(t)
        log_big_value = log_mean + 6*log_sd #6 standard deviations away from mean
        big_value = np.exp(log_big_value)
        S_max =big_value #s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
    # IF YOU WANT TO MATCH THE PAPER
    # https://www.scribd.com/document/317457984/Solution-to-Black-Scholes-P-D-E-via-Finite-Difference-Methods
    #S_max = 250 # make S-max = 250 to match paper
    d_x = (S_max - S_min)/spot_intervals

    M = spot_intervals
    N = time_intervals
    u = np.zeros((M+1, N+1)) # 0->M , 0->N

    if call:
        left_boundary =lambda S,T,K,R: 0
        right_boundary = lambda S,T,K,R: S - K*np.exp(-R*T)
        side_boundary = lambda S,K : max(S-K, 0)
    else:
        left_boundary =lambda S,T,K,R,: K*np.exp(-R*T)
        right_boundary = lambda S,T,K,R: 0
        side_boundary = lambda S,K : max(K-S, 0)

    #initial boundary conditions
    for i in range(N+1):        
        tau_i = i*d_tau
        u[0,i] = left_boundary(S_min, tau_i, k, r)#k*np.exp(-r*tau_i) #left boundary
        u[M, i] = right_boundary(S_max, tau_i, k, r) #0 #S_max - k*np.exp(-r*tau_i)   #right boundary S - Ke^(-r(T-t))

    #side boundary conditions
    for j in range(M+1):
        x_i = S_min +j*d_x
        u[j,N] = side_boundary(x_i, k)#max(k - x_i, 0)        #max(x_i - k, 0)

    #this is the change...
    S_j = lambda j: S_min +j*d_x
    alpha = lambda j: 0.5*d_tau*(r*S_j(j)/d_x-vol*vol*S_j(j)*S_j(j)/(d_x*d_x))
    beta = lambda j: 1.0+d_tau*(vol*vol*S_j(j)*S_j(j)/(d_x*d_x)+r)
    gamma = lambda j: -0.5*d_tau*(r*S_j(j)/d_x+vol*vol*S_j(j)*S_j(j)/(d_x*d_x))

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
        #u_next = LuNoPivSolve.solve(A, P_C)
        u_next = np.linalg.solve(A, P_C)
        u[1:M, i] = u_next




    i_low = int((s-S_min)/d_x)
    i_high = i_low + 1

    v_low = u[i_low,0]
    v_high = u[i_high, 0]

    s_low = S_min+ i_low*d_x
    s_high = S_min+ i_high*d_x

    price = ((s_high - s) * v_low + (s - s_low) * v_high) / (s_high - s_low);
    return price