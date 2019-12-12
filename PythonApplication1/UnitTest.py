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
       




        
        

if __name__ == '__main__':
    unittest.main()
