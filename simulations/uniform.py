### Implementation of EAST elimination alg for
### All epsilon good arms

import numpy as np 
from utils import *

class uniform(all_eps_bandit):
    def __init__(self, epsilon, means, noise_var, delta, gamma=0,
                                                         burn_amt=1,
                                                         maxpulls=25e6,
                                                         verbose=True):
        super().__init__(epsilon, means, noise_var, delta)
        self.maxpulls = maxpulls
        self.gamma = gamma
        self.verbose = verbose
        self.uniform = uniform      # optional for uniform sampling instead

    def is_known(self, i):
        return self.lbs[i] > self.Ut or self.ubs[i] < self.Lt

    def stopping_cond(self):
        self.Ut = max(self.ubs) - self.epsilon
        self.Lt = max(self.lbs) - self.epsilon
        return all([self.is_known(i) for i in range(self.narms)])

    def printout(self):
        if self.verbose and not self.total_pulls % 10000:
            print('Pulls: {}'.format(self.total_pulls))

    def run(self):
        '''Run uniform for all epsilon good problem'''
        while self.total_pulls < self.maxpulls:
            for arm in range(self.narms):
                self.pull(arm)
            self.printout()
            if self.stopping_cond(): break
        self.printout()
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    from time import time
    means = np.ones(20)
    means[-5:] = 0
    epsilon = 0.8
    noise_var = 1
    delta = 0.1
    kappa = 0.5
    np.random.seed(42)
    instance = uniform(epsilon, means, noise_var, delta, verbose=True,
                                                      maxpulls=25000000)
    ts = time()
    instance.run()
    print('total time: {}'.format(time() - ts))

    