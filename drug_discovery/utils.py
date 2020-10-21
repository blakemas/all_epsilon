import numpy as np 
import clean_data as cd 

class all_eps_bandit:
    def __init__(self, epsilon, delta, subset=None):
        '''
        A parent class for different algorithms to solve the "all epsilon
        good" problem. Algorithm is responsible for choosing which arm to 
        pull and having a method that can return a guess at the set of 
        all epsilon good arms. This class has methods to compute precision,
        recall, and F score based on the predicted set. Inheritting class is
        responsible for tracking these values and deciding when to save them.

        epsilon: double - all arms within epsilon of best
        delta: delta in finite LIL bound
        subset: list -  a subset of indices to search over
        '''
        self.epsilon = epsilon
        self.means, self.names = cd.get_data()
        if subset is not None:
            self.means = self.means[subset]
            self.names = self.names[subset]
        self.narms = self.means.shape[0]     # number of arms
        self.noise_var = 1  
        self.noise_sig = np.sqrt(self.noise_var)
        # self.bnd_eps = bnd_eps
        self.delta = delta
        self.get_true_eps_good()
        self.emps = np.zeros(self.means.shape)       # empirical means
        self.ubs = np.inf*np.ones(self.means.shape)        # upper bounds
        self.lbs = -np.inf*np.ones(self.means.shape)        # lower bounds
        self.pulls = np.zeros(self.means.shape)      # number of pulls of each
        self.total_pulls = 0

    def pull(self, arm, union=True):
        # pull an arm and update its bounds and empirical mean
        reward = self.means[arm] + np.random.normal(scale=self.noise_sig)
        self.update_emp(arm, reward)
        self.update_bounds(arm, union=union)

    def update_emp(self, arm, reward):
        total_prev_reward = self.emps[arm] * self.pulls[arm]
        self.pulls[arm] += 1    # count new pull
        self.total_pulls += 1
        self.emps[arm] = (total_prev_reward + reward) / self.pulls[arm]

    def update_bounds(self, arm, union=True):
        width = self.bound(self.pulls[arm], union=union)
        self.ubs[arm] = min(self.ubs[arm], self.emps[arm] + width)
        self.lbs[arm] = max(self.lbs[arm], self.emps[arm] - width)

    def bound(self, pulls, union=True):
        '''Compute LIL based bound with #pulls samples'''
        if union:
            return self.noise_sig * \
                    np.sqrt(2.08*np.log(self.narms * np.log2(2*pulls) /
                        self.delta) / pulls)
        else:
            return self.noise_sig * \
                    np.sqrt(2.08*np.log(np.log2(2*pulls)/self.delta) / pulls)

    def burn_in(self, amount=1, union=True):
        '''Pull all arms a fixed amount of times'''
        for _ in range(amount):
            for arm in range(self.narms): 
                self.pull(arm, union=union)

    def get_true_eps_good(self):
        best = max(self.means)
        self.eps_good_arms = set([i for i in range(self.narms) 
                                if self.means[i] >= best - self.epsilon])
        self.true_bad_arms = set([i for i in range(self.narms) 
                                if self.means[i] < best - self.epsilon])

    def precision(self, output, compare=None):
        '''Compute fraction of output guesses that are epsilon good'''
        if not len(output): return 1
        if compare is None: compare = self.eps_good_arms
        n_true_pos = len(output.intersection(compare))
        total_pred = len(output)
        return n_true_pos / total_pred

    def recall(self, output, compare=None):
        if not output: return 0
        if compare is None: compare = self.eps_good_arms
        n_true_pos = len(output.intersection(compare))
        total_pos = len(compare)
        return n_true_pos / total_pos

    def f_score(self, output, compare=None):
        p = self.precision(output, compare=compare)
        r = self.recall(output, compare=compare)
        if not p and not r: return 0
        return 2*p*r / (p + r)


def display_bounds(instance):
    import matplotlib.pyplot as plt 
    thresh = max(instance.means) - instance.epsilon
    if hasattr(instance, 'multiplicative'):
        thresh = max(instance.means) * (1 - instance.epsilon)
    if hasattr(instance, 'linear'):
        thresh = max(instance.means) * (1 - instance.epsilon) - instance.eta
    plt.figure(2)
    plt.plot(instance.emps, 'b', label='empirical', marker="o")
    plt.plot(instance.means, 'k', label='true', marker="o")
    plt.plot(instance.lbs, 'g', marker=11)
    plt.plot(instance.ubs, 'r', marker=10)
    plt.hlines(thresh, 0, len(instance.means)-1, label='Threshold')
    plt.legend(loc='best')
    plt.xlabel('Arm')
    plt.ylabel('Mean')
    plt.title('Final upper and lower bounds')
    # plt.savefig('too_early.pdf')
    plt.show()

def compute_alpha_beta(means, thresh):
    means = sorted(means)[::-1]
    above = [m for m in means if m >= thresh]
    below = [m for m in means if m < thresh]
    alpha = above[-1] - thresh 
    beta  = thresh - below[0]
    return alpha, beta 