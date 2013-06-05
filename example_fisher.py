import numpy as np
from scipy.stats import binom as binomial_distribution
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(0)

    c = 28
    n = 40

    p_value = 1.0 - binomial_distribution.cdf(c-1, n, 0.5)

    print "p_value =", p_value

    x = np.arange(n+1, dtype=np.int)
    plt.figure()
    plt.plot(x, binomial_distribution.pmf(x, n, 0.5), 'ko')
    plt.vlines(x, [0], binomial_distribution.pmf(x, n, 0.5)) # , 'ko')
    plt.plot(c, 0.004, 'kv', markersize=18)
    plt.xlabel("correct predictions ($c$)", fontsize=22)
    plt.ylabel("$p(c)$", fontsize=22)
    plt.title(r"Bin($c; n=40, p=\frac{1}{2}$)", fontsize=22)
    plt.savefig("binomial_pmf_c%d_n%d.pdf" % (c, n))
