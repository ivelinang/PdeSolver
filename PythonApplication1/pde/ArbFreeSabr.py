import numpy as np

from linearsolve import *

from scipy.interpolate import interp1d

from pde.Pde1dGeneric import *


def computeBoundaries(alpha, beta, nu, rho, f, T, nd):
    zmin = -nd*np.sqrt(T)
    zmax = -zmin

    if (beta <1.0):
        ybar = -f**(1.0-beta)/(1.0-beta)
        zbar = -1.0/nu*np.log((np.sqrt(1.0-rho*rho+(rho+nu*ybar/alpha)**2)-rho-nu*ybar/alpha)/(1.0-rho))

        if zbar > zmin:
            zmin = zbar

    return zmin, zmax


def Y(alpha, nu, rho, zm):
    return alpha/nu*(np.sinh(nu*zm)+rho*(np.cosh(nu*zm)-1.0))

def F(f, beta, ym):
    return (f**(1.0-beta)+(1.0-beta)*ym)**(1.0/(1.0-beta))

def C(alpha, beta, rho, nu, ym, Fm):
    return np.sqrt(alpha*alpha +2.0*rho*alpha*nu*ym + nu*nu*ym*ym)*(Fm**(beta))

def G(f, beta, Fm, j0):
    G = (Fm**beta - f**beta)/(Fm-f)
    G[j0] = beta / (f**(1.0-beta)) # to account for Fm-f = 0
    return G

def solveStep(Fm, Cm, Em, dt, h, P, PL, PR):
    frac = dt/(2.0*h)
    M = len(P)

    B = np.zeros(M)
    B[1:M-1] = 1.0 + frac*(Cm[1:M-1] * Em[1:M-1] * (1.0/(Fm[2:M]-Fm[1:M-1]) + 1.0/(Fm[1:M-1] - Fm[0:M-2])))

    C = np.zeros(M-1) 
    C[1:M-1] = -frac*Cm[2:M] * Em[2:M] / (Fm[2:M] - Fm[1:M-1])

    A = np.zeros(M-1) 
    A[0:M-2]= -frac*Cm[0:M-2]*Em[0:M-2]/(Fm[1:M-1] - Fm[:M-2])

    B[0] = Cm[0] / (Fm[1] - Fm[0])*Em[0]
    C[0] = Cm[1] / (Fm[1]-Fm[0])*Em[1]

    B[M-1] = Cm[M-1]/(Fm[M-1] - Fm[M-2])*Em[M-1]
    A[M-2] = Cm[M-2] / (Fm[M-1] - Fm[M-2])*Em[M-2]
    # diagonal + lower + upper
    tri = np.diag(B) + np.diag(A, -1) + np.diag(C, 1)
    P[0] = 0
    P[M-1]=0
    # solve the matrix
    P = np.linalg.solve(tri, P)
    PL = PL + dt*Cm[1] / (Fm[1] - Fm[0])*Em[1]*P[1]
    PR = PR + dt*Cm[M-2]/(Fm[M-1] - Fm[M-2])*Em[M-2]*P[M-2]

    return P, PL, PR





def makeTransformedDensity(alpha, beta, nu, rho, f, T, N, timesteps, nd):
    zmin, zmax = computeBoundaries(alpha, beta, nu, rho, f, T, nd)
    J = N-2
    h0 = (zmax - zmin)/J
    j0 = int(round((0-zmin)/h0))
    h = (0.0 - zmin)/(float(j0) - 0.5)
    z = np.arange(J+2)
    z = z*h + zmin
    #for i in range(len(z)):
    #    z[i] = i*h+zmin

    zmax = z[J+1]
    zm = z-0.5*h #vector
    ym = Y(alpha, nu, rho, zm) #vector
    ymax = Y(alpha, nu, rho, zmax)
    ymin = Y(alpha, nu, rho, zmin)

    Fm = F(f, beta, ym) #vector
    Fmax = F(f, beta, ymax)
    Fmin = F(f, beta, ymin)

    Fm[0] = 2.0*Fmin - Fm[1]
    Fm[J+1] = 2.0*Fmax- Fm[J]
    Cm = C(alpha, beta, rho, nu, ym, Fm)
    Cm[0] = Cm[1]
    Cm[J+1] = Cm[J]
    Gamma = G(f, beta, Fm, j0)
    dt = T/timesteps
    b = 1.0 - np.sqrt(2.0)/2.0
    dt1 = dt*b
    dt2 = dt*(1.0-2.0*b)
    Em = np.ones(J+2)
    Emdt1 = np.exp(rho*nu*alpha*Gamma*dt1)
    Emdt1[0] = Emdt1[1]
    Emdt1[J+1] = Emdt1[J]

    Emdt2 = np.exp(rho*nu*alpha*Gamma*dt2)
    Emdt2[0] = Emdt2[1]
    Emdt2[J+1] = Emdt2[J]

    PL = 0.0
    PR = 0.0
    P = np.zeros(J+2)
    P[j0] = 1.0/h

    for t in range(1, timesteps+1):
        Em = Em * Emdt1
        P1, PL1, PR1 = solveStep(Fm, Cm, Em, dt1, h, P, PL, PR)

        Em = Em * Emdt1
        P2, PL2, PR2 = solveStep(Fm, Cm, Em, dt1, h, P1, PL1, PR1)

        P = (np.sqrt(2.0) + 1.0)*P2 - np.sqrt(2)*P1
        PL = (np.sqrt(2.0) + 1.0)*PL2 - np.sqrt(2)*PL1
        PR = (np.sqrt(2.0) + 1.0)*PR2 - np.sqrt(2)*PR1
        Em = Em*Emdt2


    return P, PL, PR, zm, zmin, zmax, h




def YofF(F,  f,  beta):
	oneMinusBeta = 1.0 - beta;
	return (F**oneMinusBeta - f**oneMinusBeta) / oneMinusBeta;

def YofF_2(F,  f,  beta, alpha):
	oneMinusBeta = 1.0 - beta;
	return (F**oneMinusBeta - f**oneMinusBeta) / oneMinusBeta/alpha;


def DSqrOfF( F,  f,  beta,  alpha,  nu,  rho):
	yF = YofF(F, f, beta);
	F2beta = F**(2.0*beta);
	temp = alpha * alpha + 2.0*alpha*rho*nu*yF + nu * nu + yF * yF;
	return temp * F2beta;


def DSqrOfF_2( F,  f,  beta,  alpha,  nu,  rho):
	yF = YofF_2(F, f, beta, alpha);
	F2beta = F**(2.0*beta);
	temp = 1.0 * 1.0 + 2.0*rho*nu*yF + nu * nu + yF * yF;
	return temp * F2beta;


def GofF( F,  f,  beta):
    if F==f:
        return beta / (f**(1.0-beta))
    return ((F**beta) - (f**beta)) / (F - f);


def EofF(F,  f,  beta,  alpha,  nu,  rho,  Texp):
	G = GofF(F, f, beta);
	temp = rho * nu*alpha*G*Texp;
	return np.exp(temp);


#def EofF_2(F,  f,  beta,  alpha,  nu,  rho,  Texp):
#	G = GofF(F, f, beta);
#	temp = nu*alpha*G*Texp;
#	return np.exp(temp);



def MofF(F, f, beta, alpha,  nu,  rho,  Texp):
	D2 = DSqrOfF(F, f, beta, alpha, nu, rho);
	E = EofF(F, f, beta, alpha, nu, rho, Texp);
	return 0.5*D2*E;    


def MofF_2(F, f, beta, alpha,  nu,  rho,  Texp):
	D2 = DSqrOfF_2(F, f, beta, alpha, nu, rho);
	E = EofF(F, f, beta, alpha, nu, rho, Texp);
	return 0.5*alpha*alpha*D2*E;    


def priceOptionArbFreeSabr(f, strike, T, alpha, beta, nu, rho, spot_intervals, time_intervals):

    f_max = f + 10.0*alpha*np.sqrt(T)
    f_min = max(f - 5.0*alpha*np.sqrt(T),0.10)
   
    numXPoints = spot_intervals +2
    numTPoints = time_intervals  +1  

    UpperXLimit = f_max
    LowerXLimit = f_min

    dT = T/(time_intervals)
    myTPoints = np.zeros(numTPoints) # 0, 1 .... N
    for i in range(numTPoints):
        myTPoints[i] = i*dT

    dX = (f-f_min)
    dX = (UpperXLimit-LowerXLimit)/(spot_intervals)
    myXPoints = np.zeros(numXPoints) # 0, 1, 2....J+1
    for i in range(numXPoints):
        myXPoints[i] = LowerXLimit + (i-0.5)*dX #-0.5

    ##adjust dX
    #index = np.argmax(myXPoints>=f)
    #dX = (f-f_min)/(index-0.5)
    ##redo calcs
    #for i in range(numXPoints):
    #    myXPoints[i] = LowerXLimit + (i-0.5)*dX #-0.5


    leftBound = np.zeros(numTPoints)
    rightBound = np.zeros(numTPoints)
    initBound = np.zeros(numXPoints)

    for i in range(0, numTPoints):
        leftBound[i] = 0.0#f_min
        rightBound[i] = 0.0#f_max

    #find where Fi and Fi-1 such that Fi-1< f <Fi
    index = np.argmax(myXPoints>=f)
    initBound[index] = (1.0 -(myXPoints[index]-f)/(myXPoints[index] - myXPoints[index-1])) * 1.0/dX
    initBound[index-1] = (1.0 - (f - myXPoints[index-1])/(myXPoints[index] - myXPoints[index-1])) * 1.0/dX
    #for i in range(0, numXPoints):
    #    initBound[i] = side_boundary(myXPoints[i], k)

    a=np.zeros((numXPoints, numTPoints))
    b=np.zeros((numXPoints, numTPoints))
    c=np.zeros((numXPoints, numTPoints))

    for xInt in range(0, numXPoints):
        for tInt in range(0, numTPoints):

            F_j = myXPoints[xInt]
            t_j = myTPoints[tInt]
            a[xInt, tInt] = MofF(F_j, f, beta, alpha, nu, rho, myTPoints[tInt]) 
            b[xInt, tInt] = 0#r*myXPoints[xInt]*dT
            c[xInt, tInt] = 0#r*dT
      

    myGrid = np.zeros((numXPoints, numTPoints)) # 0->M , 0->N

    theta=0.5

    Pde1DGenericSolver2(myGrid, numXPoints, numTPoints, leftBound, rightBound, initBound, a, b, c, dX, dT, theta)

    for tInt in range(numTPoints):
        #myGrid[0, tInt] = -myGrid[1, tInt]
        #myGrid[-1, tInt] = -myGrid[-2, tInt]
        adadawdawd=2

    qL = np.zeros(numTPoints)
    qR = np.zeros(numTPoints)
    qL[0] = 0
    qR[0] = 0

    for tInt in range(0, numTPoints-1):

        #qL[tInt+1] = qL[tInt] + dT/(2.0*dX)*(a[1,tInt+1]*myGrid[1,tInt+1] - a[0,tInt+1]*myGrid[0,tInt+1] + a[1,tInt]*myGrid[1,tInt] - a[0,tInt]*myGrid[0,tInt])
        #qR[tInt+1] = qR[tInt] - dT/(2.0*dX)*(a[-1,tInt+1]*myGrid[-1,tInt+1] - a[-2,tInt+1]*myGrid[-2,tInt+1] + a[-1,tInt]*myGrid[-1,tInt] - a[-2,tInt]*myGrid[-2,tInt])

        qL[tInt+1] = qL[tInt] + dT/(dX)*(a[1,tInt+1]*myGrid[1,tInt+1] + a[1,tInt]*myGrid[1,tInt] )
        qR[tInt+1] = qR[tInt] + dT/(dX)*(a[-2,tInt+1]*myGrid[-2,tInt+1] + a[-2,tInt]*myGrid[-2,tInt] )


    awdawd=3
    qmL = qL[-1]
    midQ = sum(myGrid[1:-1,-1])*dX
    qmR = qR[-1]
    k = -(f_min - strike)/dX+1
    k = int(k)
    indexK = np.argmax(myXPoints>=strike)
    vCall1 = 0.5*myGrid[k, -1]*((f_min+k*dX-strike)**2)
    vCall2 = [(myXPoints[x]-strike)*dX*myGrid[x, -1] for x in range(k+1,numXPoints-1)]
    vCall3 = (f_max-strike)*qmR

    vCall4 = [myGrid[x, -1]*dX* max(myXPoints[x]-strike,0) for x in range(1,numXPoints-1)]

    vCall= vCall1+ sum(vCall2) + vCall3
    return vCall




    














