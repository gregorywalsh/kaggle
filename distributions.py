import numpy as np
from scipy.stats import uniform


class loguniform:
    def __init__(self, a=1, b=10, base=10):
        self.uni_loc = np.log(a) / np.log(base)
        self.uni_scale = (np.log(b) / np.log(base)) - self.uni_loc
        self.base = base

    def rvs(self, size=None, random_state=None):
        uniform_dist = uniform(loc=self.uni_loc, scale=self.uni_scale)
        if size is None:
            return np.power(self.base, uniform_dist.rvs(random_state=random_state))
        else:
            return np.power(self.base, uniform_dist.rvs(size=size, random_state=random_state))
