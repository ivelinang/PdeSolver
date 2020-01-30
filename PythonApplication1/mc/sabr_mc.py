import numpy as np
import scipy as sc

class SabrMonteCarlo(object):

    def __init__(self, fwd, alpha, beta, nu, rho, gamma, t):
        self.fwd = fwd
        self.alpha=alpha
        self.beta = beta
        self.nu = nu
        self.rho = rho
        self.gamma = gamma
        self.t = t

    def generateBM(self, rhoMtrx, nAssets, nSteps):

        X = np.random.normal(size=(nSteps, nAssets))
        C = sc.linalg.cholesky(rhoMtrx, lower=False) #this fails with negative correl! wth?
        #C = np.linalg.cholesky(rhoMtrx)
        Y = np.dot(X,C)
        return Y

    def generateBM_(self, rho, nSteps):

        X = np.random.normal(size=(2, nSteps))
        #C = sc.linalg.cholesky(rhoMtrx, lower=False)
        #C = np.linalg.cholesky(rhoMtrx)
        Y = X[0,:]*rho + np.sqrt(1.0-rho*rho)*X[1,:]
        Z = np.stack((X[0,:], Y))
        #Y = np.dot(X,C)
        return Z

    def generateForwardMC(self, bm, deltaT, f0, sigma, sigma0):
        forward = np.zeros(len(bm))
        for i in range(len(bm)):
            f_prev = f0 if i==0 else forward[i-1]
            sigma_prev = sigma0 if i==0 else sigma[i-1]
            forward[i] = f_prev + sigma_prev*f_prev**(self.beta)*bm[i]*np.sqrt(deltaT)
        return forward

    def generateForwardMC_mixed(self, bm, deltaT, f0, sigma, sigma0):
        forward = np.zeros(len(bm))
        for i in range(len(bm)):
            f_prev = f0 if i==0 else forward[i-1]
            sigma_prev = sigma0 if i==0 else sigma[i-1]
            forward[i] = f_prev + sigma_prev*f_prev**(self.beta)*bm[i]*np.sqrt(deltaT)
            forward[i] = max(forward[i], 0.0001);
        return forward

    def generateForwardMC_log(self, bm, deltaT, f0, sigma, sigma0):
        forward = np.zeros(len(bm))
        for i in range(len(bm)):
            f_prev = f0 if i==0 else forward[i-1]
            sigma_prev = sigma0 if i==0 else sigma[i-1]
            forward[i] = f_prev*np.exp(sigma_prev*np.power(f_prev,(self.beta-1.0))*bm[i]*np.sqrt(deltaT) - 0.5*sigma_prev*sigma_prev*np.power(f_prev,(2.0*self.beta-2.0))*deltaT)
        return forward

    def generateSigmaMC(self, bm, deltaT, sigma0):
        sigma = np.zeros(len(bm))
        for i in range(len(bm)):
            sigma_prev = sigma0 if i==0 else sigma[i-1]
            sigma[i] = sigma_prev*np.exp(-0.5*self.nu*self.nu*deltaT + self.nu*bm[i]*np.sqrt(deltaT))
        return sigma

    def simulate(self, deltaT):
        rhoMtrx = np.array([1.0, self.rho, self.rho, 1.0])
        rhoMtrx.shape=(2,2)
        numBM = int(self.t/deltaT)
        bm = self.generateBM(rhoMtrx, 2, numBM)
        #bm = self.generateBM_(self.rho, numBM).T
        bmSigma = bm[:,0]
        bmForward = bm[:,1]
        sigmaMC = self.generateSigmaMC(bmSigma, deltaT, self.alpha)
        forwardMC = self.generateForwardMC(bmForward, deltaT, self.fwd, sigmaMC, self.alpha)
        return forwardMC

    def simulate_log(self, deltaT):
        rhoMtrx = np.array([1.0, self.rho, self.rho, 1.0])
        rhoMtrx.shape=(2,2)
        numBM = int(self.t/deltaT)
        bm = self.generateBM(rhoMtrx, 2, numBM)
        bmSigma = bm[:,0]
        bmForward = bm[:,1]
        sigmaMC = self.generateSigmaMC(bmSigma, deltaT, self.alpha)
        forwardMC = self.generateForwardMC_log(bmForward, deltaT, self.fwd, sigmaMC, self.alpha)
        return forwardMC

    def simulate_mixed(self, deltaT):
        rhoMtrx = np.array([1.0, self.rho, self.rho, 1.0])
        rhoMtrx.shape=(2,2)
        numBM = int(self.t/deltaT)
        bm = self.generateBM(rhoMtrx, 2, numBM)
        bmSigma = bm[:,0]
        bmForward = bm[:,1]
        sigmaMC = self.generateSigmaMC(bmSigma, deltaT, self.alpha)
        forwardMC = self.generateForwardMC_mixed(bmForward, deltaT, self.fwd, sigmaMC, self.alpha)
        return forwardMC

    def priceOption(self, deltaT, nSimulations, strike, cp):
        forwardMC = [self.simulate(deltaT) for i in range(nSimulations)]
        forwardMC_ = np.stack(forwardMC)
        price = sum([max(cp*(s-strike),0) for s in forwardMC_[:, -1]])/nSimulations
        return price

    def priceOption_log(self, deltaT, nSimulations, strike, cp):
        forwardMC = [self.simulate_log(deltaT) for i in range(nSimulations)]
        forwardMC_ = np.stack(forwardMC)
        price = sum([max(cp*(s-strike),0) for s in forwardMC_[:, -1]])/nSimulations
        return price

    def priceOption_mixed(self, deltaT, nSimulations, strike, cp):
        forwardMC = [self.simulate_mixed(deltaT) for i in range(nSimulations)]
        forwardMC_ = np.stack(forwardMC)
        price = sum([max(cp*(s-strike),0) for s in forwardMC_[:, -1]])/nSimulations
        return price
