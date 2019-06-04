from payoff import *

class VanillaOption(object):

    def __init__(self, payoff:PayOff, K, r, T, sigma):
        self.payoff = payoff
        self.K = K
        self.r = r
        self.T =T
        self.sigma = sigma


