import numpy as np



def solve_pde_fe_generic(f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals, call=True, F_max=None):

    #define the parameters for the intervals
    numXUpperStdDevs = 12.0;
    numXLowerStdDevs = 12.0;
    numYStdDevs = 5.0;
    numXPoints = spot_intervals;
    numYPoints = vol_intervals;
    numTPoints = time_intervals;

    time = t;
    vol0 = alpha/(f**(1.0-beta)); #why?
    xi = nu
    strike = k
    cp = 1 if call else 0

    UpperXLimit = f + numXUpperStdDevs * vol0*sqrt(time);
    LowerXLimit = f - numXUpperStdDevs * vol0*sqrt(time);

    UpperYLimit = log(vol0) - 0.5*xi*xi*time + numYStdDevs * xi*sqrt(time);
    UpperYLimit = np.exp(UpperYLimit)

    LowerYLimit = log(vol0) - 0.5*xi*xi*time - numYStdDevs * xi*sqrt(time);
    LowerYLimit = np.exp(LowerYLimit)

    dT = time/(numTPoints-1)
    myTPoints = np.zeros(numTPoints)
    for i in range(len(myTPoints)):
        myTPoints[i] = i*dT

    dX = (UpperXLimit-LowerXLimit)/(numXPoints-1)
    myXPoints = np.zeros(numXPoints)
    for i in range(len(myXPoints)):
        myXPoints[i] = LowerXLimit + i*dX


    dY = (UpperYLimit-LowerYLimit)/(numYPoints-1)
    myYPoints = np.zeros(numYPoints)
    for i in range(len(myYPoints)):
        myYPoints[i] = LowerYLimit + i*dY


    myGrid = np.zeros([myTPoints, myXPoints, myYPoints])

    #now set initial & boundary values

    #set the Final Time T (Initial Boundary Values)
    for i in range(myXPoints):
        for j in range(myYPoints):
            fwd = myXPoints[i]
            myGrid[myTPoints-1, i, j] = max(cp*(fwd-strike), 0.0)

    #set Boundary values for the end of X values
    
    #along the y axis
    for tx in range(myTPoints):
        for y in range(myYPoints):
            # this is when F-> infinity
            fwd_max = myXPoints[-1]
            myGrid[tx, myXPoints-1, y] = max(cp*(fwd_max-strike), 0.0)
            # this is when F-> 0 (or some negative)
            fwd_min = myXPoints[0]
            myGrid[tx, 0, y] = max(cp*(fwd_min-strike), 0.0)

    #along the x axis
    for tx in range(myTPoints):
        for x in range(myXPoints):
            # this is when vol-> infinity 
            fwd = myXPoints[x]           
            myGrid[tx, x, myYPoints-1] = fwd
            # this is when vol-> 0            
            myGrid[tx, x, 0] = max(cp*(fwd-strike), 0.0)










