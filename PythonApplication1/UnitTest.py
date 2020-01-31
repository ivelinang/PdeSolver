import unittest
from PDEsolver import *
from decomposer import *
from linearsolve import *
from pdesolverfactory import *
from blackscholes import *
from forwardeuler import *
from pde.pde import *
from pde.sabr_pde import *
from mc.sabr_mc import *
from pde.Pde1dGeneric import *
from pde.ArbFreeSabr import *

import scipy as sc
import scipy.linalg as la

class Test_Tridag(unittest.TestCase):

    def test_A(self):
        a=np.array([0,6,2,4])
        b=np.array([2,3,5,3])
        c=np.array([3,9,2])
        r=np.array([21,69,34,22])
        n=4
        sol = Tridag.solve(a,b,c,r, n)
        true_sol = np.array([3,5,4,2])
        res = np.allclose(sol, true_sol)
        self.assertTrue(res)


    def test_B(self):
        A = np.array([[2,3,0,0],
                        [6,3,9,0],
                        [0,2,5,2],
                        [0,0,4,3]])
        r=np.array([21,69,34,22])        
        sol = Tridag.solve_matrix(A,r)
        true_sol = np.array([3,5,4,2])
        res = np.allclose(sol, true_sol)
        #another solve
        Qvec=np.linalg.solve(A,r)
        res2 = np.allclose(Qvec, true_sol)
        self.assertTrue(res)
        self.assertTrue(res2)

    def test_C(self):
        A = np.array([[2,3,0,0],
                        [6,3,9,0],
                        [0,2,5,2],
                        [0,0,4,3]])
        r=np.array([21,69,34,22])        
        sol = LuNoPivSolve.solve(A,r)
        true_sol = np.array([3,5,4,2])        
        res = np.allclose(sol, true_sol)
        
        self.assertTrue(res)


    def test_D(self):
        A = np.array([[1,-4],
                        [3,2],
                        [0,5],
                        ])
        r=np.array([31,9,-30])        
        #Qvec=np.linalg.solve(A,r)
        #sol = np.dot(np.linalg.inv(A), b)
        #true_sol = np.array([7, -6])
        #res = np.allclose(sol, true_sol)
        
        self.assertTrue(True)

    def test_F(self):
        A = np.matrix([[2.0,-1.0,0.0],
                        [-1.0,2.0,-1.0],
                        [0.0,-1.0,2.0]
                        ])

        
        x =  np.all(np.linalg.eigvals(A) > 0)
        self.assertTrue(x)

        sol_2 = np.linalg.cholesky(A).T
        sol_3 = la.cholesky(A)

        sol = cholesky_general(A)

        res = np.allclose(sol, sol_2)
        
        self.assertTrue(res)

    def test_E(self):
        A = np.array([[2,3,0,0],
                        [6,3,9,0],
                        [0,2,5,2],
                        [0,0,4,3]])
        r=np.array([21,69,34,22])        
        sol = LuRowPivSolve.solve(A,r)
        true_sol = np.array([3,5,4,2])
        res = np.allclose(sol, true_sol)
        
        self.assertTrue(res)

    def test_G(self):
        A = np.array([[2,3,0,0],
                        [6,3,9,0],
                        [0,2,5,2],
                        [0,0,4,3]])
        r=np.array([21,69,34,22])  
        
        #cant solve this with cholesky as matrix is not positive definite!!!
        
        self.assertTrue(True)

    def test_I(self):
        A = np.array([[2,3,0,0],
                        [6,3,9,0],
                        [0,2,5,2],
                        [0,0,4,3]])

        p,l,u = np_sol = la.lu(A)
        l_1, u_1 = lu_no_pivoting(A)
        p_2, l_2, u_2 = lu_row_pivoting(A)

        D = np.diag(np.diag(u))
        U = u/ np.diag(u)[:, None] 
        P = p.dot(l.dot(D.dot(U)))

        res = np.allclose(l, l_1)
        res2 = np.allclose(l, l_2)
        res3 = np.allclose(u, u_1)
        res4 = np.allclose(u, u_2)

        self.assertTrue(True)



class Test_PdeSolver(unittest.TestCase):

    def test_call_fe_european(self):


        s = 41.0
        k = 40.0
        vol = 0.35
        t = 0.75
        r = 0.04
        q = 0.0

        solver = make_euro_call_fe(s, k, vol, t, r, q)

        price_1 = solver.compute_price_alpha(0.1, 50)        
        price_3 = solve_pde_fe(s, k, vol, t, r, 0.1, 50)

        price_bs = BS_premium(s,k,t,r,vol)

        bool = np.isclose(price_1, price_bs, rtol= 0.0001)

              
        self.assertAlmostEquals(price_1, price_bs, delta=0.001*price_bs)
        self.assertAlmostEquals(price_3, price_bs, delta=0.001*price_bs)  


    def test_call_be_european(self):


        s = 41.0
        k = 40.0
        vol = 0.35
        t = 0.75
        r = 0.04
        q = 0.0

        solver = make_euro_call_be_lu(s, k, vol, t, r, q)

        price_1 = solver.compute_price_alpha(0.1, 50)  
        price_2 = solve_pde_be(s, k, vol, t, r, 0.1, 50)
        price_3 = solve_pde_fe(s, k, vol, t, r, 0.1, 50)
        #price_4 = solve_pde_cn(s, k, vol, t, r, 0.1, 50)

        price_bs = BS_premium(s,k,t,r,vol)

        bool = np.isclose(price_1, price_bs, rtol= 0.0001)

        self.assertAlmostEquals(price_3, price_bs, delta=0.01*price_bs)  
        self.assertAlmostEquals(price_2, price_bs, delta=0.01*price_bs)                
        self.assertAlmostEquals(price_1, price_bs, delta=0.01*price_bs)
        #self.assertAlmostEquals(price_4, price_bs, delta=0.01*price_bs)
        #self.assertAlmostEquals(price_1, price_4, delta=0.01*price_4)
        self.assertAlmostEquals(price_1, price_2, delta=0.001*price_2)


    def test_call_cn_european(self):

        s = 41.0
        k = 40.0
        vol = 0.35
        t = 0.75
        r = 0.04
        q = 0.0

        solver = make_euro_call_cn_lu(s, k, vol, t, r, q)

        price_1 = solver.compute_price_alpha(0.1, 50)  
        price_2 = solve_pde_be(s, k, vol, t, r, 0.1, 50)
        price_3 = solve_pde_fe(s, k, vol, t, r, 0.1, 50)
        price_4 = solve_pde_cn(s, k, vol, t, r, 0.1, 50)

        price_bs = BS_premium(s,k,t,r,vol)

        bool = np.isclose(price_1, price_bs, rtol= 0.0001)

        self.assertAlmostEquals(price_3, price_bs, delta=0.001*price_bs)  
        self.assertAlmostEquals(price_2, price_bs, delta=0.01*price_bs)                
        self.assertAlmostEquals(price_1, price_bs, delta=0.01*price_bs)
        self.assertAlmostEquals(price_4, price_bs, delta=0.01*price_bs)
        self.assertAlmostEquals(price_1, price_4, delta=0.001*price_4)
        self.assertAlmostEquals(price_1, price_2, delta=0.01*price_2)


    def test_call_fe_european_B(self):

        s = 41.0
        k = 40.0
        vol = 0.35
        t = 0.75
        r = 0.04
        q = 0.0       

        price_1 = solve_pde_fe_generic(s,k,vol, t,r, 250, 230) 
        price_2 = solve_pde_be(s, k, vol, t, r, 0.1, 230)
        price_3 = solve_pde_fe(s, k, vol, t, r, 0.1, 230)
        price_4 = solve_pde_cn(s, k, vol, t, r, 0.1, 230)

        price_bs = BS_premium(s,k,t,r,vol)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)
        self.assertAlmostEquals(price_2, price_bs, delta=0.000001*price_bs)
        self.assertAlmostEquals(price_3, price_bs, delta=0.000001*price_bs)
        self.assertAlmostEquals(price_4, price_bs, delta=0.000001*price_bs)


    def test_put_fe_european_B(self):

        s = 138.50
        k = 110
        vol = 0.16
        t = 0.632876712
        r = 0.01
        q = 0.0       

        price_1 = solve_pde_fe_generic(s,k,vol, t,r, 250, 2310, False, S_max=250)         

        price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)

    def test_put_fe_european_Alternative(self):

        s = 138.50
        k = 110
        vol = 0.16
        t = 0.632876712
        r = 0.01
        q = 0.0       

        price_1 = solve_pde_fe_generic(s,k,vol, t,r, 250, 2310, False, S_max=250)      
        price_2 = solve_pde_fe_better(s,k,vol, t,r, 250, 2310, False, S_max=250) 

        price_bs = BS_premium(s,k,t,r,vol, False)  
        
        #self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)
        self.assertAlmostEquals(price_1, price_2, delta=0.000001*price_1)

    def test_put_be_european_C(self):

        s = 138.50
        k = 110
        vol = 0.16
        t = 0.632876712
        r = 0.01
        q = 0.0       

        price_1 = solve_pde_be_generic(s,k,vol, t,r, 100, 230, False, S_max=250)         

        price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)


    def test_put_be_european_Generic_A(self):

        s = 138.50
        k = 110
        vol = 0.16
        t = 0.632876712
        r = 0.01
        q = 0.0     
        
        theta = 0.5

        price_1 = solve_pde_bs_generic(s,k,vol, t,r, 250, 2310, theta, False, S_max=250)         

        price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)


    def test_put_be_european_Generic_B(self):

        s = 100.00
        k = 105.0
        vol = 0.20
        t = 2.0
        r = 0.06
        q = 0.00     
        
        theta = 0.5
        call = False

        tau_final = t
        spot_intervals = 200
        time_intervals = 200
        #d_tau = tau_final/time_intervals

        #S_min = 0
        #S_max = 250
        #if S_max is None:
        log_mean = np.log(s)+(-vol*vol*0.5)*t  #np.log(s)+(r-vol*vol*0.5)*t
        log_sd = vol*np.sqrt(t)
        log_big_value = log_mean + 5*log_sd #6 standard deviations away from mean
        log_low_value = log_mean - 5*log_sd
        #big_value = np.exp(log_big_value)
        S_max =log_big_value
        S_min =log_low_value
           #s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
        # IF YOU WANT TO MATCH THE PAPER
        # https://www.scribd.com/document/317457984/Solution-to-Black-Scholes-P-D-E-via-Finite-Difference-Methods
        #S_max = 250 # make S-max = 250 to match paper
        d_x = (S_max - S_min)/(spot_intervals-1)

        M = spot_intervals# = 200
        numXPoints = M
        N = time_intervals# = 200
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
            left_boundary =lambda S,T,K,R: max(S-K, 0)*np.exp(-R*T)#0
            right_boundary = lambda S,T,K,R: max(S-K, 0)*np.exp(-R*T)#S - K*np.exp(-R*T)
            side_boundary = lambda S,K : max(S-K, 0)
        else:
            left_boundary =lambda S,T,K,R,: max(K-S, 0)*np.exp(-R*T)#K*np.exp(-R*T)
            right_boundary = lambda S,T,K,R: max(K-S, 0*np.exp(-R*T))#0
            side_boundary = lambda S,K : max(K-S, 0)

        leftBound = np.zeros(numTPoints)
        rightBound = np.zeros(numTPoints)
        initBound = np.zeros(numXPoints)

        for i in range(0, numTPoints):
            leftBound[i] = left_boundary(np.exp(S_min), myTPoints[i], k, r)
            rightBound[i] = right_boundary(np.exp(S_max), myTPoints[i], k, r)

        for i in range(0, numXPoints):
            initBound[i] = side_boundary(np.exp(myXPoints[i]), k)

        #a=np.zeros(numXPoints)
        #b=np.zeros(numXPoints)
        #c=np.zeros(numXPoints)

        a=np.zeros((numXPoints, numTPoints))
        b=np.zeros((numXPoints, numTPoints))
        c=np.zeros((numXPoints, numTPoints))

        for xInt in range(0, numXPoints):
            for tInt in range(0, numTPoints):
                a[xInt, tInt] = 0.5*vol*vol
                b[xInt, tInt] = r-0.5*vol*vol
                c[xInt, tInt] = r
      

        myGrid = np.zeros((numXPoints, numTPoints)) # 0->M , 0->N

        #price_1 = solve_pde_bs_generic(s,k,vol, t,r, 250, 2310, theta, False, S_max=250)         
        Pde1DGenericSolver(myGrid, numXPoints, numTPoints, leftBound, rightBound, initBound, a, b, c, dX, dT, theta)

        f =interp1d(myXPoints, myGrid[:, 0], kind='linear')
        price_1 = f(np.log(s))

        price_bs = BS_premium(s,k,t,r,vol, call)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)

    def test_put_be_european_Generic_F(self):

        s = 100.00
        k = 105.0
        vol = 0.20
        t = 2.0
        r = 0.00
        q = 0.00     
        
        theta = 0.5
        call = False

        tau_final = t
        spot_intervals = 200
        time_intervals = 200
        #d_tau = tau_final/time_intervals

        #S_min = 0
        #S_max = 250
        #if S_max is None:
        log_mean = np.log(s)+(-vol*vol*0.5)*t  #np.log(s)+(r-vol*vol*0.5)*t
        log_sd = vol*np.sqrt(t)
        log_big_value = log_mean + 5*log_sd #6 standard deviations away from mean
        log_low_value = log_mean - 5*log_sd
        #big_value = np.exp(log_big_value)
        S_max =log_big_value
        S_min =log_low_value
           #s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
        # IF YOU WANT TO MATCH THE PAPER
        # https://www.scribd.com/document/317457984/Solution-to-Black-Scholes-P-D-E-via-Finite-Difference-Methods
        #S_max = 250 # make S-max = 250 to match paper
        d_x = (S_max - S_min)/(spot_intervals-1)

        M = spot_intervals# = 200
        numXPoints = M
        N = time_intervals# = 200
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
            left_boundary =lambda S,T,K,R: max(S-K, 0)*np.exp(-R*T)#0
            right_boundary = lambda S,T,K,R: max(S-K, 0)*np.exp(-R*T)#S - K*np.exp(-R*T)
            side_boundary = lambda S,K : max(S-K, 0)
        else:
            left_boundary =lambda S,T,K,R,: max(K-S, 0)*np.exp(-R*T)#K*np.exp(-R*T)
            right_boundary = lambda S,T,K,R: max(K-S, 0*np.exp(-R*T))#0
            side_boundary = lambda S,K : max(K-S, 0)

        leftBound = np.zeros(numTPoints)
        rightBound = np.zeros(numTPoints)
        initBound = np.zeros(numXPoints)

        for i in range(0, numTPoints):
            leftBound[i] = left_boundary(np.exp(S_min), myTPoints[i], k, r)
            rightBound[i] = right_boundary(np.exp(S_max), myTPoints[i], k, r)

        for i in range(0, numXPoints):
            initBound[i] = side_boundary(np.exp(myXPoints[i]), k)

        #a=np.zeros(numXPoints)
        #b=np.zeros(numXPoints)
        #c=np.zeros(numXPoints)

        a=np.zeros((numXPoints, numTPoints))
        b=np.zeros((numXPoints, numTPoints))
        c=np.zeros((numXPoints, numTPoints))

        for xInt in range(0, numXPoints):
            for tInt in range(0, numTPoints):
                a[xInt, tInt] = 0.5*vol*vol
                b[xInt, tInt] = r-0.5*vol*vol
                c[xInt, tInt] = r
      

        myGrid = np.zeros((numXPoints, numTPoints)) # 0->M , 0->N

        #price_1 = solve_pde_bs_generic(s,k,vol, t,r, 250, 2310, theta, False, S_max=250)         
        Pde1DGenericSolver(myGrid, numXPoints, numTPoints, leftBound, rightBound, initBound, a, b, c, dX, dT, theta)

        f =interp1d(myXPoints, myGrid[:, 0], kind='linear')
        price_1 = f(np.log(s))

        price_bs = BS_premium(s,k,t,r,vol, call)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)


    def test_put_be_european_Generic_D(self):

        s = 100.00
        k = 105.0
        vol = 0.20
        t = 2.0
        r = 0.06
        q = 0.00     
        
        theta = 0.5
        call = False

        tau_final = t
        spot_intervals = 200
        time_intervals = 200
        #d_tau = tau_final/time_intervals

        #S_min = 0
        #S_max = 250
        #if S_max is None:
        log_mean = np.log(s)+(-vol*vol*0.5)*t  #np.log(s)+(r-vol*vol*0.5)*t
        log_sd = vol*np.sqrt(t)
        log_big_value = log_mean + 5*log_sd #6 standard deviations away from mean
        log_low_value = log_mean - 5*log_sd
        #big_value = np.exp(log_big_value)
        S_max =log_big_value
        S_min =log_low_value
           #s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
        # IF YOU WANT TO MATCH THE PAPER
        # https://www.scribd.com/document/317457984/Solution-to-Black-Scholes-P-D-E-via-Finite-Difference-Methods
        #S_max = 250 # make S-max = 250 to match paper
        d_x = (S_max - S_min)/(spot_intervals-1)

        M = spot_intervals# = 200
        numXPoints = M
        N = time_intervals# = 200
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
            left_boundary =lambda S,T,K,R: max(S-K, 0)*np.exp(-R*T)#0
            right_boundary = lambda S,T,K,R: max(S-K, 0)*np.exp(-R*T)#S - K*np.exp(-R*T)
            side_boundary = lambda S,K : max(S-K, 0)
        else:
            left_boundary =lambda S,T,K,R,: max(K-S, 0)*np.exp(-R*T)#K*np.exp(-R*T)
            right_boundary = lambda S,T,K,R: max(K-S, 0*np.exp(-R*T))#0
            side_boundary = lambda S,K : max(K-S, 0)

        leftBound = np.zeros(numTPoints)
        rightBound = np.zeros(numTPoints)
        initBound = np.zeros(numXPoints)

        for i in range(0, numTPoints):
            leftBound[i] = left_boundary(np.exp(S_min), myTPoints[i], k, r)
            rightBound[i] = right_boundary(np.exp(S_max), myTPoints[i], k, r)

        for i in range(0, numXPoints):
            initBound[i] = side_boundary(np.exp(myXPoints[i]), k)

        #a=np.zeros(numXPoints)
        #b=np.zeros(numXPoints)
        #c=np.zeros(numXPoints)

        a=np.zeros((numXPoints, numTPoints))
        b=np.zeros((numXPoints, numTPoints))
        c=np.zeros((numXPoints, numTPoints))

        for xInt in range(0, numXPoints):
            for tInt in range(0, numTPoints):
                a[xInt, tInt] = 0.5*vol*vol
                b[xInt, tInt] = r-0.5*vol*vol
                c[xInt, tInt] = -r
      

        myGrid = np.zeros((numXPoints, numTPoints)) # 0->M , 0->N

        #price_1 = solve_pde_bs_generic(s,k,vol, t,r, 250, 2310, theta, False, S_max=250)         
        Pde1DGenericSolver3(myGrid, numXPoints, numTPoints, leftBound, rightBound, initBound, a, b, c, dX, dT, theta)

        f =interp1d(myXPoints, myGrid[:, -1], kind='linear')
        price_1 = f(np.log(s))

        price_bs = BS_premium(s,k,t,r,vol, call)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)


    def test_put_be_european_Generic_E(self):

        s = 100.00
        k = 105.0
        vol = 0.20
        t = 2.0
        r = 0.00
        q = 0.00     
        
        theta = 0.5
        call = False

        tau_final = t
        spot_intervals = 200
        time_intervals = 200
        #d_tau = tau_final/time_intervals

        #S_min = 0
        #S_max = 250
        #if S_max is None:
        log_mean = np.log(s)+(-vol*vol*0.5)*t  #np.log(s)+(r-vol*vol*0.5)*t
        log_sd = vol*np.sqrt(t)
        log_big_value = log_mean + 5*log_sd #6 standard deviations away from mean
        log_low_value = log_mean - 5*log_sd
        #big_value = np.exp(log_big_value)
        S_max =log_big_value
        S_min =log_low_value
           #s*np.exp(r*t) + 4*np.sqrt(s*s*np.exp(2*r*t)*(np.exp(vol*vol*t)-1)) # mean + 4 s.d.
        # IF YOU WANT TO MATCH THE PAPER
        # https://www.scribd.com/document/317457984/Solution-to-Black-Scholes-P-D-E-via-Finite-Difference-Methods
        #S_max = 250 # make S-max = 250 to match paper
        d_x = (S_max - S_min)/(spot_intervals-1)

        M = spot_intervals# = 200
        numXPoints = M
        N = time_intervals# = 200
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
            left_boundary =lambda S,T,K,R: max(S-K, 0)*np.exp(-R*T)#0
            right_boundary = lambda S,T,K,R: max(S-K, 0)*np.exp(-R*T)#S - K*np.exp(-R*T)
            side_boundary = lambda S,K : max(S-K, 0)
        else:
            left_boundary =lambda S,T,K,R,: max(K-S, 0)*np.exp(-R*T)#K*np.exp(-R*T)
            right_boundary = lambda S,T,K,R: max(K-S, 0*np.exp(-R*T))#0
            side_boundary = lambda S,K : max(K-S, 0)

        leftBound = np.zeros(numTPoints)
        rightBound = np.zeros(numTPoints)
        initBound = np.zeros(numXPoints)

        for i in range(0, numTPoints):
            leftBound[i] = left_boundary(np.exp(S_min), myTPoints[i], k, r)
            rightBound[i] = right_boundary(np.exp(S_max), myTPoints[i], k, r)

        for i in range(0, numXPoints):
            initBound[i] = side_boundary(np.exp(myXPoints[i]), k)

        #a=np.zeros(numXPoints)
        #b=np.zeros(numXPoints)
        #c=np.zeros(numXPoints)

        a=np.zeros((numXPoints, numTPoints))
        b=np.zeros((numXPoints, numTPoints))
        c=np.zeros((numXPoints, numTPoints))

        for xInt in range(0, numXPoints):
            for tInt in range(0, numTPoints):
                a[xInt, tInt] = 0.5*vol*vol
                b[xInt, tInt] = r-0.5*vol*vol
                c[xInt, tInt] = -r
      

        myGrid = np.zeros((numXPoints, numTPoints)) # 0->M , 0->N

        #price_1 = solve_pde_bs_generic(s,k,vol, t,r, 250, 2310, theta, False, S_max=250)         
        Pde1DGenericSolver3(myGrid, numXPoints, numTPoints, leftBound, rightBound, initBound, a, b, c, dX, dT, theta)

        f =interp1d(myXPoints, myGrid[:, -1], kind='linear')
        price_1 = f(np.log(s))

        price_bs = BS_premium(s,k,t,r,vol, call)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)


    def test_put_be_european_Generic_C(self):

        s = 138.50
        k = 110
        vol = 0.16
        t = 0.632876712
        r = 0.01
        q = 0.0     
        
        theta = 0.5
        call = True

        tau_final = t
        spot_intervals = 200
        time_intervals = 200
        #d_tau = tau_final/time_intervals

        S_min = 0
        S_max = 250
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

        M = spot_intervals# = 200
        numXPoints = M
        N = time_intervals# = 200
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

        leftBound = np.zeros(numTPoints)
        rightBound = np.zeros(numTPoints)
        initBound = np.zeros(numXPoints)

        for i in range(0, numTPoints):
            leftBound[i] = left_boundary(S_min, myTPoints[i], k, r)
            rightBound[i] = right_boundary(S_max, myTPoints[i], k, r)

        for i in range(0, numXPoints):
            initBound[i] = side_boundary(myXPoints[i], k)

        #a=np.zeros(numXPoints)
        #b=np.zeros(numXPoints)
        #c=np.zeros(numXPoints)

        a=np.zeros((numXPoints, numTPoints))
        b=np.zeros((numXPoints, numTPoints))
        c=np.zeros((numXPoints, numTPoints))

        for xInt in range(0, numXPoints):
            for tInt in range(0, numTPoints):
                a[xInt, tInt] = 0.5*vol*vol*myXPoints[xInt]*myXPoints[xInt]*dT
                b[xInt, tInt] = r*myXPoints[xInt]*dT
                c[xInt, tInt] = r*dT
      

        myGrid = np.zeros((numXPoints, numTPoints)) # 0->M , 0->N

        #price_1 = solve_pde_bs_generic(s,k,vol, t,r, 250, 2310, theta, False, S_max=250)         
        Pde1DGenericSolver(myGrid, numXPoints, numTPoints, leftBound, rightBound, initBound, a, b, c, dX, dT, theta)

        f =interp1d(myXPoints, myGrid[:, 0], kind='linear')
        price_1 = f(s)

        price_bs = BS_premium(s,k,t,r,vol, call)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)


    def test_call_be_european_C(self):

        s = 138.50
        k = 110
        vol = 0.16
        t = 0.632876712
        r = 0.01
        q = 0.0       

        price_1 = solve_pde_be_generic(s,k,vol, t,r, 100, 230)         

        price_bs = BS_premium(s,k,t,r,vol)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)


    def test_call_be_european_D(self):

        s = 41.0
        k = 40.0
        vol = 0.35
        t = 0.75
        r = 0.04     

        price_1 = solve_pde_be_generic(s,k,vol, t,r, 50, 23)         

        price_bs = BS_premium(s,k,t,r,vol)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)


    def test_call_cn_european_A(self):

        s = 41.0
        k = 40.0
        vol = 0.35
        t = 0.75
        r = 0.04     

        price_1 = solve_pde_cn_generic(s,k,vol, t,r, 250, 230)         

        price_bs = BS_premium(s,k,t,r,vol)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)

    def test_call_cn_european_B(self):

        s = 138.50
        k = 110
        vol = 0.16
        t = 0.632876712
        r = 0.01
        q = 0.0       

        price_1 = solve_pde_cn_generic(s,k,vol, t,r, 250, 230)         

        price_bs = BS_premium(s,k,t,r,vol)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)


    def test_put_cn_european_A(self):

        s = 138.50
        k = 110
        vol = 0.16
        t = 0.632876712
        r = 0.01
        q = 0.0       

        price_1 = solve_pde_cn_generic(s,k,vol, t,r, 250, 2310, False, S_max=250)         

        price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_bs, delta=0.000001*price_bs)



    def test_call_sabr_pde_european_A(self):

        f = 1.0
        k = 0.5
        vol = 0.2658
        nu = 0.2555
        beta = 1.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        
        #f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals

        price_1 = solve_sabr_pde_fe_generic_log(f,k,vol, beta, rho, nu, gamma, t, 400, 100, 100, True)     
        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_true, delta=0.000001*price_1)  


    def test_call_sabr_pde_european_B(self):

        f = 1.0
        k = 0.5
        vol = 0.2658
        nu = 0.2555
        beta = 1.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        
        #f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals

        price_1 = solve_sabr_pde_fe_generic(f,k,vol, beta, rho, nu, gamma, t, 400, 100, 100, True)     
        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_true, delta=0.1*price_1)  


    def test_call_sabr_pde_european_C(self):

        f = 1.0
        k = 0.5
        vol = 0.2658
        nu = 0.2555
        beta = 1.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        
        #f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals

        price_1 = solve_sabr_pde_fe_mixed(f,k,vol, beta, rho, nu, gamma, t, 400, 100, 100, True)     
        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_true, delta=0.01*price_1)  

    def test_call_sabr_pde_european_D(self):

        f = 0.05
        k = 0.01
        vol = 0.2658
        nu = 0.2555
        beta = 0.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        
        #f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals

        price_1 = solve_sabr_pde_fe_generic(f,k,vol, beta, rho, nu, gamma, t, 400, 100, 100, True)     
        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_true, delta=0.01*price_1)  


    def test_call_sabr_pde_european_E(self):

        f = 0.05
        k = 0.01
        vol = 0.2658
        nu = 0.2555
        beta = 0.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        
        #f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals

        price_1 = solve_sabr_pde_fe_generic_log(f,k,vol, beta, rho, nu, gamma, t, 400, 100, 100, True)     
        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_true, delta=0.01*price_1)  


    def test_call_sabr_pde_european_F(self):

        f = 0.05
        k = 0.01
        vol = 0.2658
        nu = 0.2555
        beta = 0.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        
        #f, k, alpha, beta, rho, nu, gamma, t, spot_intervals, vol_intervals, time_intervals

        price_1 = solve_sabr_pde_fe_mixed(f,k,vol, beta, rho, nu, gamma, t, 400, 100, 100, True)     
        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price_1, price_true, delta=0.01*price_1)  


class Test_SabrMC(unittest.TestCase):

    def test_call_sabrMC_A(self):

        f = 0.05
        k = 0.01
        vol = 0.2658
        nu = 0.2555
        beta = 0.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        nSims =1000
        deltaT = 1.0/252.0

        sabr = SabrMonteCarlo(f, vol, beta, nu, rho, gamma, t)
        price = sabr.priceOption(deltaT, nSims, k, 1.0)

        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price, price_true, delta=0.01*price)  

    def test_call_sabrMC_B(self):

        f = 0.05
        k = 0.01
        vol = 0.2658
        nu = 0.2555
        beta = 0.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        nSims =1000
        deltaT = 1.0/252.0

        sabr = SabrMonteCarlo(f, vol, beta, nu, rho, gamma, t)
        price = sabr.priceOption_log(deltaT, nSims, k, 1.0)
        #price is nan
        #cant do log term with Monte Carlo and Beta=0
        #this comes from the following
        # forward[i] = f_prev*np.exp(sigma_prev*np.power(f_prev,(self.beta-1.0))*bm[i]*np.sqrt(deltaT) - 0.5*sigma_prev*sigma_prev*np.power(f_prev,(2.0*self.beta-2.0))*deltaT)
        #so beta-1 = 0-1 = -1
        #also 2.B-2 = -2
        #when forward becomes zero, 0^-1 or 0^-2 is non defined, so you get nan

        #so you should not simulate forward in log form when Beta=0

        #similarly you should not simulate forward in log form when forward is negative... for obvious reasons
     

        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price, price_true, delta=0.01*price) 

    def test_call_sabrMC_C(self):

        f = 0.05
        k = 0.01
        vol = 0.2658
        nu = 0.2555
        beta = 1.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        nSims =1000
        deltaT = 1.0/252.0

        sabr = SabrMonteCarlo(f, vol, beta, nu, rho, gamma, t)
        price = sabr.priceOption_log(deltaT, nSims, k, 1.0)
        #price is nan
        #cant do log term with Monte Carlo and Beta=0

        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price, price_true, delta=0.01*price) 


    def test_call_sabrMC_D(self):

        f = 0.05
        k = 0.01
        vol = 0.2658
        nu = 0.2555
        beta = 1.0
        rho = -0.3305
        t = 5.0
        gamma = 1.0
        nSims =1000
        deltaT = 1.0/252.0

        sabr = SabrMonteCarlo(f, vol, beta, nu, rho, gamma, t)
        price = sabr.priceOption(deltaT, nSims, k, 1.0)
        #price is nan
        #cant do log term with Monte Carlo and Beta=0

        price_true = 0.53575928916254456

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price, price_true, delta=0.01*price) 


    
    def test_call_sabrMC_F(self):

        f = 1.00
        k = 1.00
        vol = 1.0
        nu = 1.0
        beta = 0.30
        rho = 0.90
        t = 10.0
        gamma = 1.0
        nSims =1000
        deltaT = 20.0/252.0

        sabr = SabrMonteCarlo(f, vol, beta, nu, rho, gamma, t)
        price = sabr.priceOption_mixed(deltaT, nSims, k, 1.0)
        #price is nan
        #cant do log term with Monte Carlo and Beta=0

        price_true = 0.59381207591799812

        #price_bs = BS_premium(s,k,t,r,vol, False)  
        
        self.assertAlmostEquals(price, price_true, delta=0.01*price) 


    def test_choleshy_A(self):

        nSteps = 100
        nAssets =2
        rho = -0.3305
        rhoMtrx = np.array([1.0, rho, rho, 1.0])
        rhoMtrx.shape=(2,2)

        X = np.random.normal(size=(nSteps, nAssets))

        Y = X[:,0]*rho + np.sqrt(1.0-rho*rho)*X[:,1]
        Z = np.stack((X[:,0], Y),axis=1)
        #Y = np.dot(X,C)

        C = sc.linalg.cholesky(rhoMtrx, lower=False) #this fails with negative correl! wth?
        #C = np.linalg.cholesky(rhoMtrx)
        D = np.dot(X,C)

        self.assertTrue(np.prod(Z==D))

class Test_FreeArbSabr(unittest.TestCase):

    def test_Arb_Free_Sabr_M_Func(self):
        spot = 1.00
        rd = 0.0
        rf = 0.0
        alpha = 0.35
        beta = 0.25
        nu = 1.0
        rho = -0.10
        tau = 1.0
        forward = spot
        strike = 1.00
        DF = 1.0

        m1 = MofF(1.1, spot, beta, alpha, nu, rho, 0.5)
        m2 = MofF_2(1.1, spot, beta, alpha, nu, rho, 0.5)

        self.assertAlmostEqual(m1, m2, 5)

    def test_Arb_Free_Sabr_A(self):
        spot = 1.00
        rd = 0.0
        rf = 0.0
        alpha = 0.35
        beta = 0.25
        nu = 1.0
        rho = -0.10
        tau = 1.0
        forward = spot
        strike = 1.00
        DF = 1.0

        price = priceOptionArbFreeSabr(forward, strike, tau, alpha, beta, nu, rho, 500, 100)

        self.assertEqual(price, 0.15)

    def test_Arb_Free_Sabr_B(self):
        spot = 1.00
        rd = 0.0
        rf = 0.0
        alpha = 0.35
        beta = 0.25
        nu = 1.0
        rho = -0.10
        tau = 1.0
        forward = spot
        strike = 1.00
        DF = 1.0
        N = 500
        timesteps = 5
        nd =4

        P, PL, PR, zm, zmin, zmax, h =makeTransformedDensity(alpha, beta, nu, rho, forward, tau, N, timesteps, nd)
        aaaaaar=2


    def test_Arb_Free_Sabr_D(self):

        s = 1.0
        strike = 1.0
        vol = 1.98134
        t=10
        p=BS_premium(s, strike, t, 0, vol)
        self.assertEqual(p, 0.79845)


       




        
        

if __name__ == '__main__':
    unittest.main()
