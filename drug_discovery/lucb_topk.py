# Implementation of LUCB by Kalyanakrishnan et al.
# to find a single \epsilon good arm. Takes in a
# set of arms. 

import numpy as np 
from utils import *

class lucb_topk(all_eps_bandit):
    def __init__(self, epsilon, delta, subset=None, k=1,
                                                    oracle=True,
                                                    multiplicative=False,
                                                    maxpulls=10000,
                                                    printeach=10000,
                                                    verbose=True):
        super().__init__(epsilon, delta, subset=subset)
        self.get_mult_eps_good()
        if oracle and multiplicative:
            self.k = len(self.true_mult_eps_good)
        elif oracle and not multiplicative:
            self.k = len(self.eps_good_arms)
        else: self.k = k          # top K arms to find
        self.printeach = printeach
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

    def pull_arms(self):
        a = np.argpartition(self.emps, -self.k)
        bot, top = a[:-self.k], a[-self.k:]
        # find smalled LCB in top K emp:
        self.argmin_top = top[np.argmin(self.lbs[top])]
        self.pull(self.argmin_top)
        # find highest UCB in bottom n-k emp:
        self.argmax_bot = bot[np.argmax(self.ubs[bot])]
        self.pull(self.argmax_bot)

    def stop(self):
        # stopping condition for top K
        return self.ubs[self.argmax_bot] < self.lbs[self.argmin_top]

    def compute_emp_good(self):
        maxemp = max(self.emps)
        if self.multiplicative:
            thresh = (1 - self.epsilon) * max(self.emps) 
        else: thresh = max(self.emps) - self.epsilon
        self.emp_good = set(np.flatnonzero(self.emps > thresh))

    def compute_topk(self):
        self.emp_topk = set(np.argsort(self.emps)[-self.k:])

    def compute_errs(self):
        '''Compute 4 error metrics'''
        self.compute_topk()
        # self.compute_emp_good()
        if self.multiplicative: compare = self.true_mult_eps_good
        else: compare = self.eps_good_arms
        self.f_scores.append(self.f_score(self.emp_topk, compare=compare))
        self.precision_vals.append(self.precision(self.emp_topk, compare=compare))
        self.recall_vals.append(self.recall(self.emp_topk, compare=compare))
        self.emp_correct.append(self.emp_topk == compare)

    def printout(self):
        if self.verbose:
            print('Total pulls: {}, emp_f: {:.3f}'.format(self.total_pulls,
                                                        self.f_scores[-1]))

    def was_correct(self):
        self.compute_topk()
        true_topk = set(np.argsort(self.means)[-self.k:])
        return self.emp_topk == true_topk

    def run(self):
        self.burn_in(amount=1)    # pull all arms once
        self.compute_errs()
        while self.total_pulls < self.maxpulls:
            self.pull_arms()
            if self.total_pulls % self.printeach < 2: self.printout()
            if self.total_pulls % self.recordeach < 2: self.compute_errs()
            # if self.stop(): break   
        self.correct = self.was_correct()

if __name__ == '__main__':
    epsilon = 0.8
    delta = 0.001
    # k = 748     # number of true mult eps good arms
    np.random.seed(42)
    instance = lucb_topk(epsilon, delta, maxpulls=40000,
                                                oracle=True, 
                                                multiplicative=True)
    instance.run()
    print(instance.correct, instance.total_pulls)



