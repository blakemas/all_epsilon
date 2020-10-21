import numpy as np 
import caption_contest_data as ccd

contest = 651       # the contest with the greatest number of responses

def empirical_distribution(contest):
    df = ccd.summary(contest)
    n = df.shape[0]
    dist = np.zeros((n, 3))     # [unfunny, somewhat funny, funny]
    for i in range(n):
        dist[i] = get_single_dist(df.iloc[i])
    # tmp = np.zeros((100,3))
    # tmp[:50] = dist[:50]
    # tmp[50:] = dist[-50:]
    return dist

def get_means(contest):
    means = ccd.summary(contest)['score'].to_numpy()
    # tmp = np.zeros(100)
    # tmp[:50] = means[:50]
    # tmp[50:] = means[-50:]
    return means

def get_single_dist(row):
    total = row['count']
    fun = row['funny'] / total
    some_fun = row['somewhat_funny'] / total
    unfun = row['unfunny'] / total
    return [unfun, some_fun, fun]

def sample_from_dist(all_dists, arm):
    tmp = np.random.rand()
    if tmp < all_dists[arm][0]:
        return 1        # unfunny
    elif tmp < all_dists[arm, 0] + all_dists[arm, 1]:
        return 2        # somewhat funny
    else: 
        return 3        # funny

def make_log_data(data):
    return np.array([(np.log10(i+1), d) for (i, d) in enumerate(data)])

def plot_diff_contests_log(savename=None):
    means1 = make_log_data(sorted(get_means(651))[::-1])
    means2 = make_log_data(sorted(get_means(690))[::-1])
    means3 = make_log_data(sorted(get_means(627))[::-1])       #627 worked
    # w, h = fig.figaspect(2.)
    # fig.Figure(figsize=(w, h))
    plt.figure(1)
    ax = plt.gca() #you first need to get the axis handle
    ax.set_aspect(2) #sets the height to width ratio to 1.5. 
    w = 3       # linewidth
    plt.rcParams.update({'font.size': 17})
    p1 = plt.plot(means3[:, 0], means3[:, 1], linewidth=w)
    p2 = plt.plot(means1[:, 0], means1[:, 1], linewidth=w)
    p3 = plt.plot(means2[:, 0], means2[:, 1], linewidth=w)
    plt.hlines(max(means3[:, 1])*0.8, 0, np.log10(27), linestyle='--', color=p1[0].get_color(), label='627: 0.8\u03BC1', linewidth=w)
    plt.hlines(max(means1[:, 1])*0.8, 0, np.log10(748), linestyle='--', color=p2[0].get_color(), label='651: 0.8\u03BC1', linewidth=w)
    plt.hlines(max(means2[:, 1])*0.8, 0, np.log10(46), linestyle='--', color=p3[0].get_color(), label='690: 0.8\u03BC1', linewidth=w)
    plt.vlines(np.log10(27), 1, means3[26,1], linestyle='--', color=p1[0].get_color(), linewidth=w)
    plt.vlines(np.log10(748), 1, means1[747, 1], linestyle='--', color=p2[0].get_color(), linewidth=w)
    plt.vlines(np.log10(46), 1, means2[46,1], linestyle='--', color=p3[0].get_color(), linewidth=w)
    # plt.plot(means3[:, 0], means3[:, 1], 'r', linewidth=w)
    # plt.plot(means1[:, 0], means1[:, 1], 'b', linewidth=w)
    # plt.plot(means2[:, 0], means2[:, 1], 'g', linewidth=w)
    # plt.hlines(max(means3[:, 1])*0.8, 0, np.log10(27), color='r', linestyle='--', label='627: 0.8\u03BC1', linewidth=w)
    # plt.hlines(max(means1[:, 1])*0.8, 0, np.log10(748), color='b', linestyle='--', label='651: 0.8\u03BC1', linewidth=w)
    # plt.hlines(max(means2[:, 1])*0.8, 0, np.log10(46), color='g', linestyle='--', label='690: 0.8\u03BC1', linewidth=w)
    # plt.vlines(np.log10(27), 1, means3[26,1], color='r', linestyle='--', linewidth=w)
    # plt.vlines(np.log10(748), 1, means1[747, 1], color='b', linestyle='--', linewidth=w)
    # plt.vlines(np.log10(46), 1, means2[46,1], color='g', linestyle='--',  linewidth=w)
    ticks = [0, 1, 2, 3, 4]
    labels = [r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"]
    plt.xticks(ticks, labels)
    plt.xlabel('Arm')
    plt.ylabel('Mean')
    plt.legend(loc='best')
    plt.rcParams.update({'font.size': 17})
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')
    plt.show()
    return means1, means2, means3


def plot_diff_contests(savename=None):
    means1 = sorted(get_means(651))[::-1]
    means2 = sorted(get_means(690))[::-1]
    means3 = sorted(get_means(627))[::-1]       #627 worked
    # w, h = fig.figaspect(2.)
    # fig.Figure(figsize=(w, h))
    plt.figure(1)
    ax = plt.gca() #you first need to get the axis handle
    # ax.set_aspect(3) #sets the height to width ratio to 1.5. 
    w = 3       # linewidth
    plt.rcParams.update({'font.size': 22})
    p1 = plt.semilogx(means1, linewidth=w)
    p2 = plt.semilogx(means2, linewidth=w)
    p3 = plt.semilogx(means3, linewidth=w)
    plt.hlines(max(means1)*0.8, -10, 748, color=p1[0].get_color(), linestyle='--', label='0.8\u03BC1', linewidth=w)
    plt.hlines(max(means2)*0.8, -10, 46, color=p2[0].get_color(), linestyle='--', label='0.8\u03BC1', linewidth=w)
    plt.hlines(max(means3)*0.8, -10, 27, color=p3[0].get_color(), linestyle='--', label='0.8\u03BC1', linewidth=w)
    plt.vlines(748, 0, means1[747], color=p1[0].get_color(), linestyle='--', linewidth=w)
    plt.vlines(46, 0, means2[46], color=p2[0].get_color(), linestyle='--',  linewidth=w)
    plt.vlines(27, 0, means3[26], color=p3[0].get_color(), linestyle='--', linewidth=w)
    plt.rcParams.update({'font.size': 20})
    plt.xlabel('Arm')
    plt.ylabel('Mean')
    plt.legend(loc='best')
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')
    plt.show()


def plot_means(means, savename=None):
    means = sorted(means)[::-1]
    maxmean = max(means)
    w = 3
    plt.figure(1)
    plt.rcParams.update({'font.size': 15})
    plt.plot(means, linewidth=w)
    plt.hlines(maxmean*0.9, -10, len(means), color='r', linestyle='--', label='eps=0.1', linewidth=w)
    plt.hlines(maxmean*0.85, -10, len(means), color='g', linestyle='--', label='eps=0.15', linewidth=w)
    plt.hlines(maxmean*0.8, -10, len(means), color='m', linestyle='--', label='eps=0.2', linewidth=w)
    plt.legend(loc='best')
    plt.xlabel('Arm')
    plt.ylabel('Mean')
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    import matplotlib.figure as fig
    import matplotlib.axes as axes

    means = get_means(contest)
    fname_save = './images/tny_dif_weeks.pdf'
    # plt.plot(means)
    # plt.show()
    # savename = './images/tny_means.pdf'
    # plot_means(means, savename=savename)
    c651, c690, c627 = plot_diff_contests_log(savename=None)

    # fact 1: 46 within 10% but 748 withing 20%
    # fact 2: 73 within .2 but 1479 within 0.4
    # takeaway: the issues presented herein are present when
    #            gamma < epsilon. but gamma = epsilon not good here. 
    eps_vs_k = []       # epsilon versus k
    eps_vs_quantile = []    # epsilon versus reward quantile
    contests = []       # list of contest numbers
    k_vs_eps = []       # k versus epsilon
    alpha_vs_eps = []   # quantile vs epsilon
    num_above, frac_above = [], []
    mean_score = []
    k = 300
    alpha = 0.03
    eps = 0.15
    thresh = 1.5
    f1 = lambda k, means: 1 - means[k]/max(means)
    f2 = lambda alpha, means: 1 - means[int(len(means)*alpha)]/max(means)
    f3 = lambda eps, means : sum(means > (1-eps)*max(means))
    f4 = lambda eps, means: f3(eps, means) / len(means)
    f5 = lambda thresh, means: sum(np.array(means) > thresh)
    f6 = lambda thresh, means: sum(np.array(means) > thresh) / len(means)


    # oldest, newest = 451, 691
    # for contest in range(oldest, newest+1):
    #     try:
    #         means = sorted(get_means(contest))[::-1]
    #         eps_vs_k.append(f1(k, means))
    #         eps_vs_quantile.append(f2(alpha, means))
    #         k_vs_eps.append(f3(eps, means))
    #         alpha_vs_eps.append(f4(eps, means))
    #         contests.append(contest)
    #         num_above.append(f5(thresh, means))
    #         frac_above.append(f6(thresh, means))
    #         mean_score.append(np.mean(means))
    #     except ValueError:
    #         continue






