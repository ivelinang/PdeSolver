import numpy as np

from linearsolve import *

from scipy.interpolate import interp1d

def solve_pde_bs_generic(s, k, vol, t, r, spot_intervals, time_intervals,  theta = 0.5, call=True, S_max=None):

    #0-explicit, 1-implicit, 0.5 - Crank

    #time intervals
    tau_final = t
    #d_tau = tau_final/time_intervals

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
    numXPoints = M
    N = time_intervals
    numTPoints = N
    #u = np.zeros((M+1, N+1)) # 0->M , 0->N

    UpperXLimit = S_max
    LowerXLimit = S_min


    dT = tau_final/(numTPoints-1)
    myTPoints = np.zeros(numTPoints)
    for i in range(numTPoints):
        myTPoints[i] = i*dT

    dX = (UpperXLimit-LowerXLimit)/(numXPoints-1)
    myXPoints = np.zeros(numXPoints)
    for i in range(numXPoints):
        myXPoints[i] = LowerXLimit + i*dX

    if call:
        left_boundary =lambda S,T,K,R: 0
        right_boundary = lambda S,T,K,R: S - K*np.exp(-R*T)
        side_boundary = lambda S,K : max(S-K, 0)
    else:
        left_boundary =lambda S,T,K,R,: K*np.exp(-R*T)
        right_boundary = lambda S,T,K,R: 0
        side_boundary = lambda S,K : max(K-S, 0)

    ##initial boundary conditions
    #for i in range(N+1):        
    #    tau_i = i*d_tau
    #    u[0,i] = left_boundary(S_min, tau_i, k, r)#k*np.exp(-r*tau_i) #left boundary
    #    u[M, i] = right_boundary(S_max, tau_i, k, r) #0 #S_max - k*np.exp(-r*tau_i)   #right boundary S - Ke^(-r(T-t))

    ##side boundary conditions
    #for j in range(M+1):
    #    x_i = S_min +j*d_x
    #    u[j,N] = side_boundary(x_i, k)#max(k - x_i, 0)        #max(x_i - k, 0)

    myGrid = np.zeros((numXPoints, numTPoints)) # 0->M , 0->N

    for i in range(numTPoints):
        myGrid[0,i] = left_boundary(S_min, myTPoints[i], k, r)
        myGrid[numXPoints-1,i] = right_boundary(S_max, myTPoints[i], k, r)

    for j in range(numXPoints):
        myGrid[j,numTPoints-1] = side_boundary(myXPoints[j], k)

    myRhs = np.zeros(numXPoints)
    myUpper=np.zeros(numXPoints)
    myDiag=np.zeros(numXPoints)
    myLower=np.zeros(numXPoints)
    thisSlice=np.zeros(numXPoints)
    nextSlice=np.zeros(numXPoints)

    a=np.zeros(numXPoints)
    b=np.zeros(numXPoints)
    c=np.zeros(numXPoints)

    for xInt in range(1, numXPoints):

        a[xInt] = 0.5*vol*vol*myXPoints[xInt]*myXPoints[xInt]*dT
        b[xInt] = r*myXPoints[xInt]*dT
        c[xInt] = r*dT

    
    #now do the time stepping (from end to start)
    for tx in range(numTPoints-1, 0, -1 ):        

        inv_dX = 1.0 / dX        

        inv_2dX = 0.5*inv_dX;     

        inv_dXdX = inv_dX * inv_dX;  

        #tau_i = tx*d_tau

        #myGrid[:, tx-1]
        myRhs[0] = myGrid[0, tx-1]# = left_boundary(S_min,tau_i, k, r )
        myRhs[numXPoints-1] = myGrid[numXPoints-1, tx-1] #= right_boundary(S_max, tau_i, k, r)

        myUpper[0] = myUpper[numXPoints-1] = 0.0;
        myDiag[0] = myDiag[numXPoints-1] = 1.0;
        myLower[0] = myLower[numXPoints-1] = 0.0;

        for xInt in range(1, numXPoints-1):

            V_m = myGrid[xInt-1, tx];
            V_0 = myGrid[xInt, tx];
            V_p = myGrid[xInt+1, tx];

            V_x = (V_p - V_m)*inv_2dX;
            V_xx = (V_p + V_m - 2.0*V_0)*inv_dXdX;

            myRhs[xInt] = V_0 + (1.0 - theta) * (a[xInt] * V_xx + b[xInt] * V_x - c[xInt] * V_0);

            bTheta = b[xInt] * inv_2dX * theta;
            aTheta = a[xInt] * inv_dXdX * theta;

            myUpper[xInt] = -aTheta - bTheta;
            myDiag[xInt] = 1.0 + 2.0*aTheta + c[xInt] * theta;
            myLower[xInt] = -aTheta + bTheta;

        myLhs_X = Tridag().solve(myLower, myDiag, myUpper, myRhs, numXPoints)

        for xInt in range(0, numXPoints):
            myGrid[xInt, tx-1] = myLhs_X[xInt]

    f =interp1d(myXPoints, myGrid[:, 0], kind='linear')
    return f(s)

        #nextSlice = u[:, tx]
        
