### Implementation of EAST elimination alg for
### All epsilon good arms

import numpy as np 
from utils import *
from lucb import *

class fareast(all_eps_bandit):
    def __init__(self, epsilon, means, noise_var, delta, kappa, 
                                                            east=False,
                                                            verbose=True,
                                                            gamma=0):
        self.gamma = gamma
        self.east = east
        if not self.east: delta /= 2 # good filt / bad filt union bound
        super().__init__(epsilon, means, noise_var, delta)
        self.kappa = kappa
        self.good_set = set()    # set of eps good to max precision
        self.active_set = set(range(self.narms))     # active set of arms
        self.samples_in_round = 0       # how many samples bad filter draws
        self.verbose = verbose          # whether to print or not
        if self.east: self.samples_in_round = 10000 # print each 10k rounds
        self.Ut, self.Lt = np.inf, -np.inf

    def compute_tau_k(self, k):
        delta_k = self.delta / (2 * self.narms * k**2)
        self.tau_k = int(4 * np.sqrt(self.noise_var) * 2**(2*k) \
                                        * np.log(4 / delta_k))

    def compute_thresh_bnds(self):
        self.Ut = max(self.ubs[list(self.active_set)]) - self.epsilon
        self.Lt = max(self.lbs[list(self.active_set)]) - self.epsilon

    def pull_tau_times(self, arm):
        if self.tau_k < 1000:
            noise = np.random.normal(scale=self.noise_sig, size=(self.tau_k,))
        else:
            noise = np.random.normal(scale=self.noise_sig/np.sqrt(self.tau_k), size=1)
        self.samples_in_round += self.tau_k
        self.total_pulls += self.tau_k
        return np.mean(noise + self.means[arm])

    def true_if_len_1(self, arr):   # helper function if length is 1
        return True if len(arr) == 1 else False

    def add_to_good_set(self, arm):
        if self.lbs[arm] >= self.Ut:
            self.good_set.add(arm)

    def eliminate_from_active_set(self, arm):
        if self.ubs[arm] < self.Lt:
            self.active_set.remove(arm)     # remove a bad arm
        elif arm in self.good_set and \
            self.ubs[arm] < self.Lt + self.epsilon: # max LCB
            self.active_set.remove(arm)     # remove a good arm

    def stopping_cond(self):
        cond1 = self.active_set.issubset(self.good_set)
        cond2 = self.Ut - self.Lt < self.gamma / 2
        return cond1 or cond2

    def printout(self, k):
        if self.verbose:
            unknown = self.active_set.difference(self.good_set)
            bad_remain = len(unknown.intersection(self.true_bad_arms))
            print('Round: {}, Pulls: {}, Found: {}/{}, Bad remaining: {}, unknown: {}'\
                    .format(k, self.total_pulls, len(self.good_set),\
                                             len(self.eps_good_arms),\
                                             bad_remain,
                                             len(unknown)))

    def bad_filter_init(self, k):
        self.compute_tau_k(k)       # how many samples to get from each arm
        self.samples_in_round = 0   # samples taken so far this round
        self.find_good_arm(k)       # run LUCB to find a 2^-k good arm

    def find_good_arm(self, k):
        '''Run LUCB to find a 2^-k good arm for the bad filter'''
        if not len(self.good_set): 
            G = np.arange(self.narms)   # no good arms found yet
        else: G = list(self.good_set)     # only search over good set
        if len(G) == 1: self.i_k = G[0]
        else:
            instance = lucb(2**(-k), self.means[G], self.noise_var, self.kappa)
            instance.run()
            self.i_k = G[instance.argmax_emp]
            self.samples_in_round += instance.total_pulls
            self.total_pulls += instance.total_pulls

    def bad_filter(self, k):
        self.bad_filter_init(k)
        mu_ik = self.pull_tau_times(self.i_k)   # pull result of LUCB
        for i in self.active_set.difference(self.good_set): # unknown arms
            mu_i = self.pull_tau_times(i)   # pull specific arm
            if mu_ik - mu_i >= self.epsilon + 2**(-k + 1):
                if self.verbose: print('Bad filter removal')
                self.active_set.remove(i)

    def active_arm_with_fewest_pulls(self):
        min_pulls = min(self.pulls[list(self.active_set)])
        argmin_set = [i for i in self.active_set if self.pulls[i] == min_pulls]
        self.all_active_arms_equal_samples = self.true_if_len_1(argmin_set)
        return np.random.choice(argmin_set)

    def good_filter(self):
        for _ in range(self.samples_in_round):
            self.pull(self.active_arm_with_fewest_pulls())
            # elimination step:
            if self.all_active_arms_equal_samples:
                self.compute_thresh_bnds()
                a = self.active_set.copy()  # ow set changes size in iteration
                for i in a:
                    self.add_to_good_set(i)
                    self.eliminate_from_active_set(i)
            if self.stopping_cond(): return     # stop early
        
    def run(self):
        k = 1
        while True:
            if not self.east:
                self.bad_filter(k)
                if self.stopping_cond(): break
            self.good_filter()
            if self.stopping_cond(): break
            self.printout(k)
            k += 1                      # update counter
        self.printout(k)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    means = np.ones(100)
    # means[1:-1] = 0.965
    means[-1:] = 0
    epsilon = 0.8
    # means = 0.1*np.arange(25)[::-1]
    # epsilon = 0.75
    noise_var = 1
    delta = 0.01
    kappa = 0.5
    # np.random.seed(42) 

    instance = fareast(epsilon, means, noise_var, delta, kappa, east=False)
    instance.run()