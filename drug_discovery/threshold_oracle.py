# Implementation of APT by Locatelli et al.
# and Jain + Jamieson 2018

import numpy as np 
from utils import *

class threshold(all_eps_bandit):
    def __init__(self, epsilon, delta, subset=None, k=1,
                                                    multiplicative=False,
                                                    maxpulls=10000,
                                                    printeach=10000,
                                                    verbose=True,
                                                    method='APT'):
        super().__init__(epsilon, delta, subset=subset)
        self.get_mult_eps_good()
        if multiplicative:
            self.thresh = (1 - self.epsilon) * max(self.means)
        else:
            self.thresh = max(self.means) - self.epsilon
        self.printeach = printeach
        self.method = method        
        self.maxpulls = maxpulls
        self.recordeach = 10
        self.multiplicative = multiplicative    # for computing errors
        self.f_scores = []              # list of f scores of recall set
        self.emp_correct = []           # is set of emp means correct
        self.precision_vals, self.recall_vals = [], []
        self.verbose = verbose

    def get_mult_eps_good(self):
        self.true_mult_eps_good = [i for i in range(self.narms)
                                    if self.means[i] >=\
                                     (1-self.epsilon)*max(self.means)]

    def pull_arm(self):
        if self.method == 'APT':
            scaled_emp_gaps = np.sqrt(self.pulls) * np.abs(self.emps - self.thresh)
            self.pull(np.argmin(scaled_emp_gaps))
        else:
            pass

    def compute_emp_good(self):
        self.emp_good = set(np.flatnonzero(self.emps > self.thresh))

    def compute_errs(self):
        '''Compute 4 error metrics'''
        self.compute_emp_good()
        if self.multiplicative: compare = self.true_mult_eps_good
        else: compare = self.eps_good_arms
        self.f_scores.append(self.f_score(self.emp_good, compare=compare))
        self.precision_vals.append(self.precision(self.emp_good, compare=compare))
        self.recall_vals.append(self.recall(self.emp_good, compare=compare))
        self.emp_correct.append(self.emp_good == compare)

    def printout(self):
        if self.verbose:
            print('Total pulls: {}, emp_f: {:.3f}'.format(self.total_pulls,
                                                        self.f_scores[-1]))

    def was_correct(self):
        self.compute_emp_good()
        if self.multiplicative: compare = self.true_mult_eps_good
        else: compare = self.eps_good_arms
        return self.emp_good == set(compare)

    def run(self):
        self.burn_in(amount=1)    # pull all arms once
        self.compute_errs()
        while self.total_pulls < self.maxpulls:
            self.pull_arm()
            if not self.total_pulls % self.printeach: self.printout()
            if not self.total_pulls % self.recordeach: self.compute_errs()   
        self.correct = self.was_correct()

if __name__ == '__main__':
    epsilon = 0.8
    delta = 0.001
    np.random.seed(42)
    instance = threshold(epsilon, delta, maxpulls=40000, multiplicative=True)
    instance.run()
    print(instance.correct, instance.total_pulls)



