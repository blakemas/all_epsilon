### Implementation of LUCB scheme for all epsilon good arms

import numpy as np 
from utils import *

class st2(all_eps_bandit):
    def __init__(self, epsilon, delta, burn_amt=1, maxpulls=1e5,
                                                            printeach=5000,
                                                            verbose=True,
                                                            multiplicative=False):
        super().__init__(epsilon, delta)
        self.get_mult_eps_good()
        self.maxpulls = maxpulls
        self.burn_amt = burn_amt
        self.f_scores, self.emp_correct = [], []
        self.precision_vals, self.recall_vals = [], []
        self.argmax_ucb = 0         # arm with highest UCB
        self.argmin_good = 0        # emp good arm with lowest LCB
        self.argmax_bad = 0         # emp bad arm with highest UCB
        self.printeach = printeach
        self.recordeach = 10
        self.verbose = verbose
        self.multiplicative = multiplicative

    def get_mult_eps_good(self):
        self.true_mult_eps_good = [i for i in range(self.narms)
                                    if self.means[i] >=\
                                     (1-self.epsilon)*max(self.means)]

    def arms_to_pull(self):
        ''' Computes which are the three arms to pull per round.
            NOTE: good and bad sets must be fresh.
        '''
        # arm with highest upper bound
        self.argmax_ucb = np.argmax(self.ubs)
        # Empirically good arm with lowest lcb
        good_idx = np.argmin(self.lbs[self.emp_good])
        self.argmin_good = self.emp_good[good_idx]
        # Empirically bad arm with highest ucb if exists
        if len(self.emp_bad):
            bad_idx = np.argmax(self.ubs[self.emp_bad])
            self.argmax_bad = self.emp_bad[bad_idx]
        else: self.argmax_bad = -1  # flag so we don't pull    

    def compute_thresh(self):
        '''Computes bounds on threshold'''
        if self.multiplicative:
            self.thresh_emp = (1 - self.epsilon) * max(self.emps)
            self.thresh_ub = (1 - self.epsilon) * max(self.ubs)
            self.thresh_lb = (1 - self.epsilon) * max(self.lbs)
        else:
            self.thresh_emp = max(self.emps) - self.epsilon
            self.thresh_ub = max(self.ubs) - self.epsilon
            self.thresh_lb = max(self.lbs) - self.epsilon

    def compute_sets(self):
        '''Compute sets of empirically good and bad arms'''
        self.emp_good = np.flatnonzero(self.emps >= self.thresh_emp)
        self.emp_bad = np.flatnonzero(self.emps < self.thresh_emp)

    def compute_err(self):
        '''Compute error on empirically good set'''
        if self.multiplicative: compare = self.true_mult_eps_good
        else: compare = self.eps_good_arms
        s = set(self.emp_good)
        self.f_scores.append(self.f_score(s, compare=compare))
        self.recall_vals.append(self.recall(s, compare=compare))
        self.precision_vals.append(self.precision(s, compare=compare))
        self.emp_correct.append(s == compare)

    def stopping_condition(self):
        '''Stop when lowest good lb above theshold UB and
            highest bad ub below threshold LB or hit maxpulls.
        '''
        if self.total_pulls >= self.maxpulls: return True
        # if self.argmax_bad != -1 and \
        #     self.ubs[self.argmax_bad] >= self.thresh_lb:
        #     return False
        # if self.lbs[self.argmin_good] <= self.thresh_ub: 
        #     return False
        # return True
        
    def printout(self):
        if self.multiplicative: ngood = len(self.true_mult_eps_good)
        else: ngood = len(self.eps_good_arms)
        if self.verbose:
            print('Pulls: {}, f score: {:.3f}, emp good: {}/{}'\
                    .format(self.total_pulls, self.f_scores[-1],
                        len(self.emp_good), ngood))

    def round_setup(self):
        '''Initialize arms, sets and err before round'''
        self.compute_thresh()                # compute current threshold 
        self.compute_sets()                  # compute good and bad sets
        self.arms_to_pull()                  # find put which arm to pull

    def pull_round(self):
        '''Pull three specified arms'''
        self.pull(self.argmax_ucb)
        if self.lbs[self.argmin_good] <= self.thresh_ub:
            self.pull(self.argmin_good)
        if self.argmax_bad != -1 and\
            self.ubs[self.argmax_bad] >= self.thresh_lb:
            self.pull(self.argmax_bad)  

    def run(self):
        '''Function to run LUCB all epsilon'''
        self.burn_in(amount=self.burn_amt)
        self.round_setup()
        self.compute_err()
        while not self.stopping_condition():
            self.pull_round()   # pull current arms
            self.round_setup()  # result
            if self.total_pulls % self.printeach < 3: self.printout()
            if self.total_pulls % self.recordeach < 3: self.compute_err()
        self.printout()


if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    epsilon = 0.8
    delta = 0.001
    maxpulls = 100000
    np.random.seed(42)
    instance = st2(epsilon, delta, maxpulls=maxpulls, 
                                            multiplicative=True)
    instance.run()
