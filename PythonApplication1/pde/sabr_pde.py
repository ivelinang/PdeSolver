import numpy as np

from .linearsolve import *



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
            myGrid[tx, x, myYPoints-1] = fwd if call else strike
            # this is when vol-> 0            
            myGrid[tx, x, 0] = max(cp*(fwd-strike), 0.0)




    #now do the time stepping (from end to start)
    for tx in range(myTPoints-1, -1, -1 ):        

        inv_dX = 1.0 / dX
        inv_dY = 1.0 / dY

        inv_2dX = 0.5*inv_dX;
        inv_2dY = 0.5*inv_dY;

        inv_dXdX = inv_dX * inv_dX;
        inv_dXdY = inv_dX * inv_dY;
        inv_dYdY = inv_dY * inv_dY;

        inv_4dXdY = 0.25*inv_dXdY;

        myRhs = np.zeros([myXPoints-1, myYPoints-1])
        myOper_X=np.zeros([myXPoints-1, myYPoints-1])
        myOper_Y=np.zeros([myXPoints-1, myYPoints-1])

        for x in range(1, myXPoints-1):
            for y in range(1, myYPoints-1):

                v_m0 = myGrid[tx, x-1, y]
                v_mp = myGrid[tx, x-1, y+1]
                v_mm = myGrid[tx, x-1, y-1]
                v_00 = myGrid[tx, x, y]
                v_0m = myGrid[tx, x, y-1]
                v_0p = myGrid[tx, x, y+1]
                v_p0 = myGrid[tx, x+1, y]
                v_pm = myGrid[tx, x+1, y-1]
                v_pp = myGrid[tx, x+1, y+1]

                #now do the finite difference
                v_x = (v_p0 - v_m0)*inv_2dX
                v_xx = (v_p0 + v_m0 - 2.0*v_00)*inv_dXdX
                v_y = (v_0p - v_0m)*inv_2dY
                v_yy = (v_0p + v_0m - 2.0*v_00)*inv_dYdY
                v_xy = (v_pp + v_mm - v_pm - v_mp)*inv_4dXdY

                fwd = myXPoints[x]
                vol = myYPoints[y]

                a = 0 #no v_x term
                b = 0.5*fwd**(2.0*beta)*vol*vol
                c = 0 #no v_y term
                d = 0.5*nu*nu*vol**(2.0*gamma)
                e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                myOper_Y[x, y] =(c*v_y + d*v_yy)*dT
                myOper_X[x, y] =(a*v_x + b*v_xx)*dT

                myRhs[x,y]  = v_00 + (a*v_x + b*v_xx)*dT + (c*v_y + d*v_yy)*dT + e*v_xy*dT

        #now do triadag


        #copy over values
        myV_star =np.zeros([myXPoints+1, myYPoints+1])
        for x in range(myXPoints):
            myV_star[x, 0] = myGrid[tx, x, 0]
            myV_star[x, myYPoints] = myGrid[tx, x, myYPoints]


        myUpper_X=np.zeros(myXPoints+1)
        myLower_X=np.zeros(myXPoints+1)
        myDiag_X=np.zeros(myXPoints+1)
        myTemp_X=np.zeros(myXPoints+1)

        myUpper_Y=np.zeros(myYPoints+1)
        myLower_Y=np.zeros(myYPoints+1)
        myDiag_Y=np.zeros(myYPoints+1)
        myTemp_Y=np.zeros(myYPoints+1)

        for y in range(1, myYPoints-1):

            myUpper_X[0] = 0
            myUpper_X[myXPoints] = 0
            myLower_X[0] = 0
            myLower_X[myXPoints] = 0
            myDiag_X[0] = 1.0
            myDiag_X[myXPoints] = 1.0
            myTemp_X[myXPoints] = myGrid[tx, myXPoints, y]


            for x in range(1, myXPoints-1):

                fwd = myXPoints[x]
                vol = myYPoints[y]

                a = 0 #no v_x term
                b = 0.5*fwd**(2.0*beta)*vol*vol
                #c = 0 #no v_y term
                #d = 0.5*nu*nu*vol**(2.0*gamma)
                #e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                atmp = a*0.25*dT*inv_dX #zero
                btmp = b*0.5*dT*inv_dXdX

                myUpper_X[x] = -btmp 
                myDiag_X[x] = 1.0 +2.0*btmp
                myLower_X[x] = -btmp

                myTemp_X[x] = myRhs[x,y]

            #now do the tridiag
            myLhs_X = Tridiag().solve(myLower_X, myDiag_X, myUpper_X, myTemp_X, myXPoints+1)

            for x in range(myXPoints):
                myV_star[x, y] = myLhs_X[x]

        for x in range(1, myXPoints-1):
            for y in range(1, myYPoints-1):
                myRhs[x, y] = myV_star(x,y) - 0.5*myOper_Y[x,y]

        for x in range(1, myXPoints-1):

            myUpper_Y[0] = 0
            myUpper_Y[myYPoints] = 0
            myLower_Y[0] = 0
            myLower_Y[myYPoints] = 0
            myDiag_Y[0] = 1.0
            myDiag_Y[myYPoints] = 1.0
            myTemp_Y[myXPoints] = myGrid[tx-1, myXPoints, y]


            for y in range(1, myYPoints-1):

                c = 0 #no v_y term
                d = 0.5*nu*nu*vol**(2.0*gamma)
                e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                ctmp = c*0.25*dT*inv_dY #zero
                dtmp = d*0.5*dT*inv_dYdY

                myUpper_Y[Y] = -dtmp 
                myDiag_Y[y] = 1.0 +2.0*dtmp
                myLower_Y[Y] = -dtmp

                myTemp_X[y] = myRhs[x,y]

            myLhs_Y = Tridiag().solve(myLower_y, myDiag_Y, myUpper_Y, myTemp_Y, myYPoints+1)

            for y in range(myYPoints):
                Grid[tx-1,x, y] = myLhs_Y[y]



















