#import numpy as np

class PayOff(object):

    def __init__(self, strike):
        self.strike = strike

    def get(self, spot):
        pass


class PayOffCall(PayOff):

    def __init__(self, strike):
        super().__init__(strike)

    def get(self, spot):
        return max(spot - self.strike, 0)


class PayOffPut(PayOff):

    def __init__(self, strike):
        super().__init__(strike)

    def get(self, spot):
        return max(self.strike - strike, 0)