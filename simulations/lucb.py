# Implementation of LUCB by Kalyanakrishnan et al.
# to find a single \epsilon good arm. Takes in a
# set of arms. 

import numpy as np 
from utils import *

class lucb(all_eps_bandit):
    def __init__(self, epsilon, means, noise_var, delta):
        super().__init__(epsilon, means, noise_var, delta)
        self.argmax_emp, self.argmax_other = 0, 0

    def pull_arms(self):
        # pull highest emp mean
        self.argmax_emp = np.argmax(self.emps)
        self.pull(self.argmax_emp, union=False)
        # pull highest other UCB
        others = [i for i in range(self.narms) if i != self.argmax_emp]
        self.argmax_other = others[np.argmax(self.ubs[others])]
        self.pull(self.argmax_other, union=False)

    def stop(self):
        lb_emp = self.lbs[self.argmax_emp]
        ub_other = self.ubs[self.argmax_other]
        return (ub_other - lb_emp) <= self.epsilon

    def run(self):
        self.burn_in(amount=1, union=False)    # pull all arms once
        while True:
            self.pull_arms()
            if self.stop(): break   
        self.correct = self.argmax_emp in self.eps_good_arms

if __name__ == '__main__':
    means = np.zeros(100)
    means[:3] = 1
    epsilon = 0.3
    noise_var = 1
    delta = 0.4
    instance = lucb(epsilon, means, noise_var, delta)
    instance.run()
    print(instance.correct, instance.argmax_emp, instance.total_pulls)



