import numpy as np
from scipy.stats import norm

def BS_premium(s0, k, T, r, sigma, call=True):
    d1 = (np.log(s0/k)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if call is  True:
        sign = 1
    else:
        sign = -1
    premium = sign*s0*norm.cdf(sign*d1)-sign*k*np.exp(-r*T)*norm.cdf(sign*d2)
    return premium  
