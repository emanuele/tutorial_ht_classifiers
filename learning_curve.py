import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_distribution

def credible_interval(c, n, CI=0.95, alpha=1, beta=1):
    lower_tail = (1.0 - CI) / 2.0
    upper_tail = 1.0 - lower_tail
    x = np.linspace(0,1,1000)
    a = c+alpha
    b = n-c+beta
    mean = a / (a + b)
    cdf = beta_distribution.cdf(x, a, b)
    lower_bound = x[cdf > lower_tail][0]
    upper_bound = x[cdf < upper_tail][-1]
    return mean, lower_bound, upper_bound
                    

if __name__ == '__main__':

    np.random.seed(0)

    N = 100

    m = np.arange(1,N+1, 5 , dtype=np.float)

    learning_curve = (1.0 - 1.0/(m**0.2)) * 0.5 + 0.5
    # learning_curve = 1.0 / (1.0 + np.exp(-(m**0.2)))

    mean_lb_ub = []
    for i, n in enumerate(N-m):
        c = np.round(learning_curve[i] * n)
        mean_lb_ub.append(credible_interval(c,n))

    mean_lb_ub = np.array(mean_lb_ub)
    # learning_curve = mean_lb_ub[:,0]
    lb = learning_curve - mean_lb_ub[:,1]
    ub = mean_lb_ub[:,2] - learning_curve
    
    plt.figure()
    plt.plot(m, learning_curve, 'ko')
    plt.xlabel("$m = |D_{train}|$", fontsize='large')
    plt.ylabel("accuracy")
    plt.title("Learning Curve ($|D|=100$)", fontsize='large')
    plt.ylim([0.0, 1.0])
    plt.savefig('learning_curve.pdf')

    plt.figure()
    plt.errorbar(m, learning_curve, yerr=[lb, ub], fmt='ko')
    plt.xlabel("$m = |D_{train}|$", fontsize='large')
    plt.ylabel("accuracy")
    plt.title("Learning Curve ($|D|=100$, $95\%$ CI)")
    plt.ylim([0.0, 1.0])
    plt.savefig('learning_curve_with_ci.pdf')

    p = np.linspace(0,1,500)
    alpha = 1
    beta = 1
    c = 28
    n = 40
    plt.figure()
    plt.plot(p, beta_distribution.pdf(p, c + alpha, n - c + beta), 'k-')
    plt.xlabel("$p$", fontsize=20)
    plt.ylabel("$p(p|c=%d,n=%d)$" % (c,n), fontsize=20)
    plt.title("Posterior of $p$ under Uniform$[0,1]$ prior ($c=%d,n=%d,\\alpha=%d,\\beta=%d$)" % (c,n,alpha,beta))
    plt.savefig("p_posterior_uniform_c%d_n%d.pdf" % (c,n))
