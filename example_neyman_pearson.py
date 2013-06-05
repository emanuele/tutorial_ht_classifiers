import numpy as np
from scipy.stats import binom as binomial_distribution

def ppv(alpha, beta, R):
    """The Positive Predictive Value, i.e. p(H_1 true | H_0 rejected).
    """
    return ((1.0 - beta) * R) / ((1.0 - beta) * R + alpha)


if __name__ == '__main__':

    np.random.seed(0)

    n = 40
    p0 = 0.5
    p1 = 0.65
    alpha_threshold = 0.05

    x = np.arange(n+1, dtype=np.int)

    # p_c_given_H0 = binomial_distribution.pmf(x, n, p0)
    alpha = 1.0 - binomial_distribution.cdf(x-1, n, p0)

    # p_given_H1 = binomial_distribution.pmf(x, n, p1)
    beta = binomial_distribution.cdf(x-1, n, p1)

    alpha_threshold_actual = alpha[alpha<=alpha_threshold][0]
    x_threshold = x[alpha<=alpha_threshold][0]
    beta_actual = beta[alpha<=alpha_threshold][0]
    print alpha_threshold_actual, x_threshold, beta_actual

    for threshold in range(20,30):
        print threshold, alpha[x==threshold], beta[x==threshold]

    plt.figure()
    plt.plot(x, binomial_distribution.pmf(x, n, p0), 'bo')
    plt.vlines(x, [0], binomial_distribution.pmf(x, n, p0), color='b')
    plt.plot(x, binomial_distribution.pmf(x, n, p1), 'ro')
    plt.vlines(x, [0], binomial_distribution.pmf(x, n, p1), color='r')
    x_above = x[x >= x_threshold]
    x_below = x[x < x_threshold]
    plt.vlines(x_above, [0], binomial_distribution.pmf(x_above, n, p0), color='b', linewidth=4)
    plt.vlines(x_below, [0], binomial_distribution.pmf(x_below, n, p1), color='r', linewidth=4)

    plt.xlabel("correct predictions ($c$)", fontsize=22)
    plt.ylabel("$p(c)$", fontsize=22)

    plt.text(12,0.1, "$H_0$", fontsize=28, color='b')
    plt.text(32,0.1, "$H_1$", fontsize=28, color='r')
    plt.savefig("np_alpha_beta_p%0.2f_R%d.pdf" % (p1,x_threshold))


    plt.figure()
    alpha = 0.05
    beta = np.linspace(0,1,50)
    R = 0.2
    ppv_example = ppv(alpha=alpha, beta=beta, R=R)
    plt.plot(beta, ppv_example, 'k-', label="PPV")
    plt.xlabel(r'$\beta$', fontsize=20)
    plt.ylabel('PPV')
    plt.title(r"The Positive Predictive Value ($\alpha$=%0.2f, $R$=%0.2f)" % (alpha, R))
