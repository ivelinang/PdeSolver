import numpy as np

from linearsolve import *

from scipy.interpolate import RectBivariateSpline, interp2d



def solve_sabr_pde_fe_generic_log(f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals, call=True, F_max=None):

    #define the parameters for the intervals
    numXUpperStdDevs = 12.0;
    numXLowerStdDevs = 12.0;
    numYStdDevs = 5.0;
    numXPoints = spot_intervals;
    numYPoints = vol_intervals;
    numTPoints = time_intervals;

    numXPoints_m1 = spot_intervals-1;
    numYPoints_m1 = vol_intervals-1;

    time = t;
    vol0 = alpha #alpha/(f**(1.0-beta)); #why?
    xi = nu
    strike = k
    cp = 1.0 if call else 0.0   
    
    #UpperXLimit=np.zeros(numTPoints)
    #LowerXLimit=np.zeros(numTPoints)
    #UpperYLimit=np.zeros(numTPoints)
    #LowerYLimit=np.zeros(numTPoints) 

    #for tx in range(numTPoints):
    #    tt = myTPoints[tx]
    #    UpperXLimit[tx] = np.log(f) - 0.5*vol0*vol0*tt+ numXUpperStdDevs * vol0*np.sqrt(tt);
    #    LowerXLimit[tx] = np.log(f) - 0.5*vol0*vol0*tt - numXUpperStdDevs * vol0*np.sqrt(tt);

    #    UpperYLimit[tx] = np.log(vol0) - 0.5*xi*xi*tt + numYStdDevs * xi*np.sqrt(tt);
    #    #UpperYLimit = np.exp(UpperYLimit)

    #    LowerYLimit[tx] = np.log(vol0) - 0.5*xi*xi*tt - numYStdDevs * xi*np.sqrt(tt);
    #    #LowerYLimit = np.exp(LowerYLimit)

    UpperXLimit = np.log(f) - 0.5*vol0*vol0*time+ numXUpperStdDevs * vol0*np.sqrt(time);
    LowerXLimit = np.log(f) - 0.5*vol0*vol0*time - numXUpperStdDevs * vol0*np.sqrt(time);

    UpperYLimit = np.log(vol0) - 0.5*xi*xi*time + numYStdDevs * xi*np.sqrt(time);
    #UpperYLimit = np.exp(UpperYLimit)

    LowerYLimit = np.log(vol0) - 0.5*xi*xi*time - numYStdDevs * xi*np.sqrt(time);
    #LowerYLimit = np.exp(LowerYLimit)

    dT = time/(numTPoints-1)
    myTPoints = np.zeros(numTPoints)
    for i in range(numTPoints):
        myTPoints[i] = i*dT

    dX = (UpperXLimit-LowerXLimit)/(numXPoints-1)
    myXPoints = np.zeros(numXPoints)
    for i in range(numXPoints):
        myXPoints[i] = LowerXLimit + i*dX


    dY = (UpperYLimit-LowerYLimit)/(numYPoints-1)
    myYPoints = np.zeros(numYPoints)
    for i in range(numYPoints):
        myYPoints[i] = LowerYLimit + i*dY

    


    myGrid = np.zeros([numTPoints, numXPoints, numYPoints])

    #now set initial & boundary values

    #set the Final Time T (Initial Boundary Values)
    for i in range(numXPoints):
        for j in range(numYPoints):
            fwd = myXPoints[i]
            myGrid[numTPoints-1, i, j] = max(cp*(np.exp(fwd)-strike), 0.0)

    #set Boundary values for the end of X values
    
    #along the y axis
    for tx in range(numTPoints):
        for y in range(numYPoints):
            # this is when F-> infinity
            fwd_max = UpperXLimit
            myGrid[tx, numXPoints_m1, y] = max(cp*(np.exp(fwd_max)-strike), 0.0)
            # this is when F-> 0 (or some negative)
            fwd_min = LowerXLimit#myXPoints[0]
            myGrid[tx, 0, y] = max(cp*(np.exp(fwd_min)-strike), 0.0)

    #along the x axis
    for tx in range(numTPoints):
        for x in range(numXPoints):
            # this is when vol-> infinity 
            fwd = np.exp(myXPoints[x])           
            myGrid[tx, x, numYPoints_m1] = 0# fwd if call else strike
            # this is when vol-> 0            
            myGrid[tx, x, 0] = 0 # max(cp*(fwd-strike), 0.0)




    #now do the time stepping (from end to start)
    for tx in range(numTPoints-1, 0, -1 ):        

        inv_dX = 1.0 / dX
        inv_dY = 1.0 / dY

        inv_2dX = 0.5*inv_dX;
        inv_2dY = 0.5*inv_dY;

        inv_dXdX = inv_dX * inv_dX;
        inv_dXdY = inv_dX * inv_dY;
        inv_dYdY = inv_dY * inv_dY;

        inv_4dXdY = 0.25*inv_dXdY;

        myRhs = np.zeros([numXPoints_m1, numYPoints_m1])
        myOper_X=np.zeros([numXPoints_m1, numYPoints_m1])
        myOper_Y=np.zeros([numXPoints_m1, numYPoints_m1])

        for x in range(1, numXPoints_m1):
            for y in range(1, numYPoints_m1):

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

                fwd = np.exp(myXPoints[x])
                vol = np.exp(myYPoints[y])

                a = -0.5*fwd**(2.0*beta-2.0)*vol*vol #no v_x term
                b = 0.5*fwd**(2.0*beta-2.0)*vol*vol
                c = -0.5*nu*nu*vol**(2.0*gamma-2.0) #no v_y term
                d = 0.5*nu*nu*vol**(2.0*gamma-2.0)
                e = fwd**(beta-1.0)*nu*vol**(gamma)*rho

                myOper_Y[x, y] =(c*v_y + d*v_yy)*dT
                myOper_X[x, y] =(a*v_x + b*v_xx)*dT*0.5  #half_dt??

                myRhs[x,y]  = v_00 + (a*v_x + b*v_xx)*dT*0.5 + (c*v_y + d*v_yy)*dT + e*v_xy*dT

        #now do triadag


        #copy over values
        myV_star =np.zeros([numXPoints, numYPoints])
        for x in range(numXPoints):
            myV_star[x, 0] = myGrid[tx, x, 0]
            myV_star[x, numYPoints_m1] = myGrid[tx, x, numYPoints_m1]


        myUpper_X=np.zeros(numXPoints)
        myLower_X=np.zeros(numXPoints)
        myDiag_X=np.zeros(numXPoints)
        myTemp_X=np.zeros(numXPoints)

        myUpper_Y=np.zeros(numYPoints)
        myLower_Y=np.zeros(numYPoints)
        myDiag_Y=np.zeros(numYPoints)
        myTemp_Y=np.zeros(numYPoints)

        for y in range(1, numYPoints_m1):

            myUpper_X[0] = 0
            myUpper_X[numXPoints_m1] = 0
            myLower_X[0] = 0
            myLower_X[numXPoints_m1] = 0
            myDiag_X[0] = 1.0
            myDiag_X[numXPoints_m1] = 1.0
            myTemp_X[0] = myGrid[tx,0, y]
            myTemp_X[numXPoints_m1] = myGrid[tx, numXPoints_m1, y]


            for x in range(1, numXPoints_m1):

                fwd = np.exp(myXPoints[x])
                vol = np.exp(myYPoints[y])

                #a = 0 #no v_x term
                #b = 0.5*fwd**(2.0*beta)*vol*vol
                a = -0.5*fwd**(2.0*beta-2.0)*vol*vol #no v_x term
                b = 0.5*fwd**(2.0*beta-2.0)*vol*vol
                #c = 0 #no v_y term
                #d = 0.5*nu*nu*vol**(2.0*gamma)
                #e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                atmp = a*0.25*dT*inv_dX #zero
                btmp = b*0.5*dT*inv_dXdX

                myUpper_X[x] = -atmp-btmp 
                myDiag_X[x] = 1.0 +2.0*btmp #+0.5*r*dT
                myLower_X[x] = +atmp-btmp

                myTemp_X[x] = myRhs[x,y]

            #now do the tridiag
            myLhs_X = Tridag().solve(myLower_X, myDiag_X, myUpper_X, myTemp_X, numXPoints)

            #lets try with np.solve to improve speed
            #set up matrix A
            #A = np.zeros((numXPoints, numXPoints))
            #for row in range(numXPoints):
            #    A[row, row] = myDiag_X[row]
            #for row in range(numXPoints-1):
            #    A[row, row+1] =  myUpper_X[row] #gamma(row+1)
            #for row in range(numXPoints-1):
            #    A[row, row-1] = myLower_X[row]

            #C = np.zeros(numXPoints)
            #for row in range(numXPoints):
            #    C[row] = myTemp_X[row]

            #myLhs_X = np.linalg.solve(A, C)

            for x in range(numXPoints):
                myV_star[x, y] = myLhs_X[x]

        for x in range(1, numXPoints_m1):
            for y in range(1, numYPoints_m1):
                myRhs[x, y] = myV_star[x,y] - 0.5*myOper_Y[x,y]

        for x in range(1, numXPoints_m1):

            myUpper_Y[0] = 0
            myUpper_Y[numYPoints_m1] = 0
            myLower_Y[0] = 0
            myLower_Y[numYPoints_m1] = 0
            myDiag_Y[0] = 1.0
            myDiag_Y[numYPoints_m1] = 1.0
            myTemp_Y[0] = myGrid[tx-1, x, 0]
            myTemp_Y[numYPoints_m1] = myGrid[tx-1, x, numYPoints_m1]


            for y in range(1, numYPoints_m1):

                fwd = np.exp(myXPoints[x])
                vol = np.exp(myYPoints[y])

                #c = 0 #no v_y term
                #d = 0.5*nu*nu*vol**(2.0*gamma)
                c = -0.5*nu*nu*vol**(2.0*gamma-2.0) #no v_y term
                d = 0.5*nu*nu*vol**(2.0*gamma-2.0)
                #e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                ctmp = c*0.25*dT*inv_dY #zero
                dtmp = d*0.5*dT*inv_dYdY

                myUpper_Y[y] = -ctmp -dtmp 
                myDiag_Y[y] = 1.0 +2.0*dtmp
                myLower_Y[y] = +ctmp-dtmp

                myTemp_Y[y] = myRhs[x,y]

            myLhs_Y = Tridag().solve(myLower_Y, myDiag_Y, myUpper_Y, myTemp_Y, numYPoints)

            #A = np.zeros((numYPoints, numYPoints))
            #for row in range(numYPoints):
            #    A[row, row] = myDiag_Y[row]
            #for row in range(numYPoints-1):
            #    A[row, row+1] =  myUpper_Y[row] #gamma(row+1)
            #for row in range(numYPoints-1):
            #    A[row, row-1] = myLower_Y[row]

            #C = np.zeros(numYPoints)
            #for row in range(numYPoints):
            #    C[row] = myTemp_Y[row]

            #myLhs_Y = np.linalg.solve(A, C)

            for y in range(1, numYPoints_m1):
                myGrid[tx-1,x, y] = myLhs_Y[y]
    
    myXYValues = myGrid[0, :, :].T #reshape((numYPoints, numXPoints))
    f_i = interp2d(myXPoints, myYPoints, myXYValues, kind='cubic')

    return f_i(np.log(f), np.log(vol0))



def solve_sabr_pde_fe_generic(f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals, call=True, F_max=None):

    #define the parameters for the intervals
    numXUpperStdDevs = 12.0;
    numXLowerStdDevs = 12.0;
    numYStdDevs = 5.0;
    numXPoints = spot_intervals;
    numYPoints = vol_intervals;
    numTPoints = time_intervals;

    numXPoints_m1 = spot_intervals-1;
    numYPoints_m1 = vol_intervals-1;

    time = t;
    vol0 = alpha #alpha/(f**(1.0-beta)); #why?
    xi = nu
    strike = k
    cp = 1.0 if call else 0.0   
    
    #UpperXLimit=np.zeros(numTPoints)
    #LowerXLimit=np.zeros(numTPoints)
    #UpperYLimit=np.zeros(numTPoints)
    #LowerYLimit=np.zeros(numTPoints) 

    #for tx in range(numTPoints):
    #    tt = myTPoints[tx]
    #    UpperXLimit[tx] = np.log(f) - 0.5*vol0*vol0*tt+ numXUpperStdDevs * vol0*np.sqrt(tt);
    #    LowerXLimit[tx] = np.log(f) - 0.5*vol0*vol0*tt - numXUpperStdDevs * vol0*np.sqrt(tt);

    #    UpperYLimit[tx] = np.log(vol0) - 0.5*xi*xi*tt + numYStdDevs * xi*np.sqrt(tt);
    #    #UpperYLimit = np.exp(UpperYLimit)

    #    LowerYLimit[tx] = np.log(vol0) - 0.5*xi*xi*tt - numYStdDevs * xi*np.sqrt(tt);
    #    #LowerYLimit = np.exp(LowerYLimit)

    UpperXLimit = f + numXUpperStdDevs * vol0*np.sqrt(time);
    #UpperXLimit = np.exp(UpperXLimit)
    LowerXLimit = f - numXUpperStdDevs * vol0*np.sqrt(time);
    #LowerXLimit = np.exp(LowerXLimit)

    UpperYLimit = vol0  + numYStdDevs * xi*np.sqrt(time);
    #UpperYLimit = np.exp(UpperYLimit)
    LowerYLimit = vol0 - numYStdDevs * xi*np.sqrt(time);
    #LowerYLimit = np.exp(LowerYLimit)

    dT = time/(numTPoints-1)
    myTPoints = np.zeros(numTPoints)
    for i in range(numTPoints):
        myTPoints[i] = i*dT

    dX = (UpperXLimit-LowerXLimit)/(numXPoints-1)
    myXPoints = np.zeros(numXPoints)
    for i in range(numXPoints):
        myXPoints[i] = LowerXLimit + i*dX


    dY = (UpperYLimit-LowerYLimit)/(numYPoints-1)
    myYPoints = np.zeros(numYPoints)
    for i in range(numYPoints):
        myYPoints[i] = LowerYLimit + i*dY

    


    myGrid = np.zeros([numTPoints, numXPoints, numYPoints])

    #now set initial & boundary values

    #set the Final Time T (Initial Boundary Values)
    for i in range(numXPoints):
        for j in range(numYPoints):
            fwd = myXPoints[i]
            myGrid[numTPoints-1, i, j] = max(cp*((fwd)-strike), 0.0)

    #set Boundary values for the end of X values
    
    #along the y axis
    for tx in range(numTPoints):
        for y in range(numYPoints):
            # this is when F-> infinity
            fwd_max = UpperXLimit
            myGrid[tx, numXPoints_m1, y] = max(cp*((fwd_max)-strike), 0.0)
            # this is when F-> 0 (or some negative)
            fwd_min = LowerXLimit#myXPoints[0]
            myGrid[tx, 0, y] = max(cp*((fwd_min)-strike), 0.0)

    #along the x axis
    for tx in range(numTPoints):
        for x in range(numXPoints):
            # this is when vol-> infinity 
            fwd = myXPoints[x]         
            myGrid[tx, x, numYPoints_m1] = 0#fwd if call else strike
            # this is when vol-> 0            
            myGrid[tx, x, 0] = 0#max(cp*(fwd-strike), 0.0)




    #now do the time stepping (from end to start)
    for tx in range(numTPoints-1, 0, -1 ):        

        inv_dX = 1.0 / dX
        inv_dY = 1.0 / dY

        inv_2dX = 0.5*inv_dX;
        inv_2dY = 0.5*inv_dY;

        inv_dXdX = inv_dX * inv_dX;
        inv_dXdY = inv_dX * inv_dY;
        inv_dYdY = inv_dY * inv_dY;

        inv_4dXdY = 0.25*inv_dXdY;

        myRhs = np.zeros([numXPoints_m1, numYPoints_m1])
        myOper_X=np.zeros([numXPoints_m1, numYPoints_m1])
        myOper_Y=np.zeros([numXPoints_m1, numYPoints_m1])

        for x in range(1, numXPoints_m1):
            for y in range(1, numYPoints_m1):

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

                a =0 #-0.5*fwd**(2.0*beta-2.0)*vol*vol #no v_x term
                b = 0.5*fwd**(2.0*beta)*vol*vol
                c =0 #-0.5*nu*nu*vol**(2.0*gamma-2.0) #no v_y term
                d = 0.5*nu*nu*vol**(2.0*gamma)
                e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                myOper_Y[x, y] =(c*v_y + d*v_yy)*dT
                myOper_X[x, y] =(a*v_x + b*v_xx)*dT*0.5  #half_dt??

                myRhs[x,y]  = v_00 + (a*v_x + b*v_xx)*dT*0.5 + (c*v_y + d*v_yy)*dT + e*v_xy*dT

        #now do triadag


        #copy over values
        myV_star =np.zeros([numXPoints, numYPoints])
        for x in range(numXPoints):
            myV_star[x, 0] = myGrid[tx, x, 0]
            myV_star[x, numYPoints_m1] = myGrid[tx, x, numYPoints_m1]


        myUpper_X=np.zeros(numXPoints)
        myLower_X=np.zeros(numXPoints)
        myDiag_X=np.zeros(numXPoints)
        myTemp_X=np.zeros(numXPoints)

        myUpper_Y=np.zeros(numYPoints)
        myLower_Y=np.zeros(numYPoints)
        myDiag_Y=np.zeros(numYPoints)
        myTemp_Y=np.zeros(numYPoints)

        for y in range(1, numYPoints_m1):

            myUpper_X[0] = 0
            myUpper_X[numXPoints_m1] = 0
            myLower_X[0] = 0
            myLower_X[numXPoints_m1] = 0
            myDiag_X[0] = 1.0
            myDiag_X[numXPoints_m1] = 1.0
            myTemp_X[0] = myGrid[tx,0, y]
            myTemp_X[numXPoints_m1] = myGrid[tx, numXPoints_m1, y]


            for x in range(1, numXPoints_m1):

                fwd = myXPoints[x]
                vol = myYPoints[y]

                #a = 0 #no v_x term
                #b = 0.5*fwd**(2.0*beta)*vol*vol
                a =0 #-0.5*fwd**(2.0*beta-2.0)*vol*vol #no v_x term
                b = 0.5*fwd**(2.0*beta)*vol*vol
                #c = 0 #no v_y term
                #d = 0.5*nu*nu*vol**(2.0*gamma)
                #e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                atmp = a*0.25*dT*inv_dX #zero
                btmp = b*0.5*dT*inv_dXdX

                myUpper_X[x] = -atmp-btmp 
                myDiag_X[x] = 1.0 +2.0*btmp #+0.5*r*dT
                myLower_X[x] = +atmp-btmp

                myTemp_X[x] = myRhs[x,y]

            #now do the tridiag
            myLhs_X = Tridag().solve(myLower_X, myDiag_X, myUpper_X, myTemp_X, numXPoints)

            #lets try with np.solve to improve speed
            #set up matrix A
            #A = np.zeros((numXPoints, numXPoints))
            #for row in range(numXPoints):
            #    A[row, row] = myDiag_X[row]
            #for row in range(numXPoints-1):
            #    A[row, row+1] =  myUpper_X[row] #gamma(row+1)
            #for row in range(numXPoints-1):
            #    A[row, row-1] = myLower_X[row]

            #C = np.zeros(numXPoints)
            #for row in range(numXPoints):
            #    C[row] = myTemp_X[row]

            #myLhs_X = np.linalg.solve(A, C)

            for x in range(numXPoints):
                myV_star[x, y] = myLhs_X[x]

        for x in range(1, numXPoints_m1):
            for y in range(1, numYPoints_m1):
                myRhs[x, y] = myV_star[x,y] - 0.5*myOper_Y[x,y]

        for x in range(1, numXPoints_m1):

            myUpper_Y[0] = 0
            myUpper_Y[numYPoints_m1] = 0
            myLower_Y[0] = 0
            myLower_Y[numYPoints_m1] = 0
            myDiag_Y[0] = 1.0
            myDiag_Y[numYPoints_m1] = 1.0
            myTemp_Y[0] = myGrid[tx-1, x, 0]
            myTemp_Y[numYPoints_m1] = myGrid[tx-1, x, numYPoints_m1]


            for y in range(1, numYPoints_m1):

                fwd = myXPoints[x]
                vol = myYPoints[y]

                #c = 0 #no v_y term
                #d = 0.5*nu*nu*vol**(2.0*gamma)
                c =0 #-0.5*nu*nu*vol**(2.0*gamma-2.0) #no v_y term
                d = 0.5*nu*nu*vol**(2.0*gamma)
                #e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                ctmp = c*0.25*dT*inv_dY #zero
                dtmp = d*0.5*dT*inv_dYdY

                myUpper_Y[y] = -ctmp -dtmp 
                myDiag_Y[y] = 1.0 +2.0*dtmp
                myLower_Y[y] = +ctmp-dtmp

                myTemp_Y[y] = myRhs[x,y]

            myLhs_Y = Tridag().solve(myLower_Y, myDiag_Y, myUpper_Y, myTemp_Y, numYPoints)

            #A = np.zeros((numYPoints, numYPoints))
            #for row in range(numYPoints):
            #    A[row, row] = myDiag_Y[row]
            #for row in range(numYPoints-1):
            #    A[row, row+1] =  myUpper_Y[row] #gamma(row+1)
            #for row in range(numYPoints-1):
            #    A[row, row-1] = myLower_Y[row]

            #C = np.zeros(numYPoints)
            #for row in range(numYPoints):
            #    C[row] = myTemp_Y[row]

            #myLhs_Y = np.linalg.solve(A, C)

            for y in range(1, numYPoints_m1):
                myGrid[tx-1,x, y] = myLhs_Y[y]
    
    myXYValues = myGrid[0, :, :].T #reshape((numYPoints, numXPoints))
    f_i = interp2d(myXPoints, myYPoints, myXYValues, kind='cubic')

    return f_i(f, vol0)[0]


def solve_sabr_pde_fe_mixed(f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals, call=True, F_max=None):

    assert(beta == 0.0, "beta must be ==1.0")

    #define the parameters for the intervals
    numXUpperStdDevs = 12.0;
    numXLowerStdDevs = 12.0;
    numYStdDevs = 5.0;
    numXPoints = spot_intervals;
    numYPoints = vol_intervals;
    numTPoints = time_intervals;

    numXPoints_m1 = spot_intervals-1;
    numYPoints_m1 = vol_intervals-1;

    time = t;
    vol0 = alpha #alpha/(f**(1.0-beta)); #why?
    xi = nu
    strike = k
    cp = 1.0 if call else 0.0   
    
    #UpperXLimit=np.zeros(numTPoints)
    #LowerXLimit=np.zeros(numTPoints)
    #UpperYLimit=np.zeros(numTPoints)
    #LowerYLimit=np.zeros(numTPoints) 

    #for tx in range(numTPoints):
    #    tt = myTPoints[tx]
    #    UpperXLimit[tx] = np.log(f) - 0.5*vol0*vol0*tt+ numXUpperStdDevs * vol0*np.sqrt(tt);
    #    LowerXLimit[tx] = np.log(f) - 0.5*vol0*vol0*tt - numXUpperStdDevs * vol0*np.sqrt(tt);

    #    UpperYLimit[tx] = np.log(vol0) - 0.5*xi*xi*tt + numYStdDevs * xi*np.sqrt(tt);
    #    #UpperYLimit = np.exp(UpperYLimit)

    #    LowerYLimit[tx] = np.log(vol0) - 0.5*xi*xi*tt - numYStdDevs * xi*np.sqrt(tt);
    #    #LowerYLimit = np.exp(LowerYLimit)

    UpperXLimit = f + numXUpperStdDevs * vol0*np.sqrt(time);
    #UpperXLimit = np.exp(UpperXLimit)
    LowerXLimit = f - numXUpperStdDevs * vol0*np.sqrt(time);
    #LowerXLimit = np.exp(LowerXLimit)

    #UpperYLimit = vol0  + numYStdDevs * xi*np.sqrt(time);
    #UpperYLimit = np.exp(UpperYLimit)
    #LowerYLimit = vol0 - numYStdDevs * xi*np.sqrt(time);
    #LowerYLimit = np.exp(LowerYLimit)

    UpperYLimit = np.log(vol0) - 0.5*xi*xi*time + numYStdDevs * xi*np.sqrt(time);
    #UpperYLimit = np.exp(UpperYLimit)

    LowerYLimit = np.log(vol0) - 0.5*xi*xi*time - numYStdDevs * xi*np.sqrt(time);

    dT = time/(numTPoints-1)
    myTPoints = np.zeros(numTPoints)
    for i in range(numTPoints):
        myTPoints[i] = i*dT

    dX = (UpperXLimit-LowerXLimit)/(numXPoints-1)
    myXPoints = np.zeros(numXPoints)
    for i in range(numXPoints):
        myXPoints[i] = LowerXLimit + i*dX


    dY = (UpperYLimit-LowerYLimit)/(numYPoints-1)
    myYPoints = np.zeros(numYPoints)
    for i in range(numYPoints):
        myYPoints[i] = LowerYLimit + i*dY

    


    myGrid = np.zeros([numTPoints, numXPoints, numYPoints])

    #now set initial & boundary values

    #set the Final Time T (Initial Boundary Values)
    for i in range(numXPoints):
        for j in range(numYPoints):
            fwd = myXPoints[i]
            myGrid[numTPoints-1, i, j] = max(cp*((fwd)-strike), 0.0)

    #set Boundary values for the end of X values
    
    #along the y axis
    for tx in range(numTPoints):
        for y in range(numYPoints):
            # this is when F-> infinity
            fwd_max = UpperXLimit
            myGrid[tx, numXPoints_m1, y] = max(cp*((fwd_max)-strike), 0.0)
            # this is when F-> 0 (or some negative)
            fwd_min = LowerXLimit#myXPoints[0]
            myGrid[tx, 0, y] = max(cp*((fwd_min)-strike), 0.0)

    #along the x axis
    for tx in range(numTPoints):
        for x in range(numXPoints):
            # this is when vol-> infinity 
            fwd = myXPoints[x]         
            myGrid[tx, x, numYPoints_m1] = fwd if call else strike
            # this is when vol-> 0            
            myGrid[tx, x, 0] = max(cp*(fwd-strike), 0.0)




    #now do the time stepping (from end to start)
    for tx in range(numTPoints-1, 0, -1 ):        

        inv_dX = 1.0 / dX
        inv_dY = 1.0 / dY

        inv_2dX = 0.5*inv_dX;
        inv_2dY = 0.5*inv_dY;

        inv_dXdX = inv_dX * inv_dX;
        inv_dXdY = inv_dX * inv_dY;
        inv_dYdY = inv_dY * inv_dY;

        inv_4dXdY = 0.25*inv_dXdY;

        myRhs = np.zeros([numXPoints_m1, numYPoints_m1])
        myOper_X=np.zeros([numXPoints_m1, numYPoints_m1])
        myOper_Y=np.zeros([numXPoints_m1, numYPoints_m1])

        for x in range(1, numXPoints_m1):
            for y in range(1, numYPoints_m1):

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
                vol = np.exp( myYPoints[y])

                a =0 #-0.5*fwd**(2.0*beta-2.0)*vol*vol #no v_x term
                b = 0.5*fwd**(2.0*beta)*vol*vol
                #c =0 #-0.5*nu*nu*vol**(2.0*gamma-2.0) #no v_y term
                #d = 0.5*nu*nu*vol**(2.0*gamma)
                c = -0.5*nu*nu*vol**(2.0*gamma-2.0) #no v_y term
                d = 0.5*nu*nu*vol**(2.0*gamma-2.0)
                e = fwd**(beta)*nu*vol**(gamma)*rho

                myOper_Y[x, y] =(c*v_y + d*v_yy)*dT
                myOper_X[x, y] =(a*v_x + b*v_xx)*dT*0.5  #half_dt??

                myRhs[x,y]  = v_00 + (a*v_x + b*v_xx)*dT*0.5 + (c*v_y + d*v_yy)*dT + e*v_xy*dT

        #now do triadag


        #copy over values
        myV_star =np.zeros([numXPoints, numYPoints])
        for x in range(numXPoints):
            myV_star[x, 0] = myGrid[tx, x, 0]
            myV_star[x, numYPoints_m1] = myGrid[tx, x, numYPoints_m1]


        myUpper_X=np.zeros(numXPoints)
        myLower_X=np.zeros(numXPoints)
        myDiag_X=np.zeros(numXPoints)
        myTemp_X=np.zeros(numXPoints)

        myUpper_Y=np.zeros(numYPoints)
        myLower_Y=np.zeros(numYPoints)
        myDiag_Y=np.zeros(numYPoints)
        myTemp_Y=np.zeros(numYPoints)

        for y in range(1, numYPoints_m1):

            myUpper_X[0] = 0
            myUpper_X[numXPoints_m1] = 0
            myLower_X[0] = 0
            myLower_X[numXPoints_m1] = 0
            myDiag_X[0] = 1.0
            myDiag_X[numXPoints_m1] = 1.0
            myTemp_X[0] = myGrid[tx,0, y]
            myTemp_X[numXPoints_m1] = myGrid[tx, numXPoints_m1, y]


            for x in range(1, numXPoints_m1):

                fwd = myXPoints[x]
                vol = np.exp( myYPoints[y])

                #a = 0 #no v_x term
                #b = 0.5*fwd**(2.0*beta)*vol*vol
                a =0 #-0.5*fwd**(2.0*beta-2.0)*vol*vol #no v_x term
                b = 0.5*fwd**(2.0*beta)*vol*vol
                #c = 0 #no v_y term
                #d = 0.5*nu*nu*vol**(2.0*gamma)
                #e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                atmp = a*0.25*dT*inv_dX #zero
                btmp = b*0.5*dT*inv_dXdX

                myUpper_X[x] = -atmp-btmp 
                myDiag_X[x] = 1.0 +2.0*btmp #+0.5*r*dT
                myLower_X[x] = +atmp-btmp

                myTemp_X[x] = myRhs[x,y]

            #now do the tridiag
            myLhs_X = Tridag().solve(myLower_X, myDiag_X, myUpper_X, myTemp_X, numXPoints)

            #lets try with np.solve to improve speed
            #set up matrix A
            #A = np.zeros((numXPoints, numXPoints))
            #for row in range(numXPoints):
            #    A[row, row] = myDiag_X[row]
            #for row in range(numXPoints-1):
            #    A[row, row+1] =  myUpper_X[row] #gamma(row+1)
            #for row in range(numXPoints-1):
            #    A[row, row-1] = myLower_X[row]

            #C = np.zeros(numXPoints)
            #for row in range(numXPoints):
            #    C[row] = myTemp_X[row]

            #myLhs_X = np.linalg.solve(A, C)

            for x in range(numXPoints):
                myV_star[x, y] = myLhs_X[x]

        for x in range(1, numXPoints_m1):
            for y in range(1, numYPoints_m1):
                myRhs[x, y] = myV_star[x,y] - 0.5*myOper_Y[x,y]

        for x in range(1, numXPoints_m1):

            myUpper_Y[0] = 0
            myUpper_Y[numYPoints_m1] = 0
            myLower_Y[0] = 0
            myLower_Y[numYPoints_m1] = 0
            myDiag_Y[0] = 1.0
            myDiag_Y[numYPoints_m1] = 1.0
            myTemp_Y[0] = myGrid[tx-1, x, 0]
            myTemp_Y[numYPoints_m1] = myGrid[tx-1, x, numYPoints_m1]


            for y in range(1, numYPoints_m1):

                fwd = myXPoints[x]
                vol = np.exp( myYPoints[y])

                #c = 0 #no v_y term
                #d = 0.5*nu*nu*vol**(2.0*gamma)
                #c =0 #-0.5*nu*nu*vol**(2.0*gamma-2.0) #no v_y term
                #d = 0.5*nu*nu*vol**(2.0*gamma)
                c = -0.5*nu*nu*vol**(2.0*gamma-2.0) #no v_y term
                d = 0.5*nu*nu*vol**(2.0*gamma-2.0)
                #e = fwd**(beta)*nu*vol**(gamma+1.0)*rho

                ctmp = c*0.25*dT*inv_dY #zero
                dtmp = d*0.5*dT*inv_dYdY

                myUpper_Y[y] = -ctmp -dtmp 
                myDiag_Y[y] = 1.0 +2.0*dtmp
                myLower_Y[y] = +ctmp-dtmp

                myTemp_Y[y] = myRhs[x,y]

            myLhs_Y = Tridag().solve(myLower_Y, myDiag_Y, myUpper_Y, myTemp_Y, numYPoints)

            #A = np.zeros((numYPoints, numYPoints))
            #for row in range(numYPoints):
            #    A[row, row] = myDiag_Y[row]
            #for row in range(numYPoints-1):
            #    A[row, row+1] =  myUpper_Y[row] #gamma(row+1)
            #for row in range(numYPoints-1):
            #    A[row, row-1] = myLower_Y[row]

            #C = np.zeros(numYPoints)
            #for row in range(numYPoints):
            #    C[row] = myTemp_Y[row]

            #myLhs_Y = np.linalg.solve(A, C)

            for y in range(1, numYPoints_m1):
                myGrid[tx-1,x, y] = myLhs_Y[y]
    
    myXYValues = myGrid[0, :, :].T #reshape((numYPoints, numXPoints))
    f_i = interp2d(myXPoints, myYPoints, myXYValues, kind='cubic')

    return f_i(f, np.log(vol0))[0]


















