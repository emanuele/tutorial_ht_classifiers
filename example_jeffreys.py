import numpy as np
from scipy.stats import beta as beta_distribution, binom as binomial_distribution

def compute_posteriors(c, n, p_H1=0.5, iterations=1000, verbose=True):
    p_H2 = 1.0 - p_H1
    p_data_given_H1 = binomial_distribution.pmf(c, n, 0.5)
    if verbose: print "p(data|H_1) =", p_data_given_H1

    # Montecarlo:
    low = 0.5
    high = 1.0
    pi = np.random.uniform(low=0.5, high=1.0, size=iterations)
    p_data_given_H2 = (binomial_distribution.pmf(c, n, pi) * 1.0 / (high-low)).mean()
    if verbose: print "p(data|H_2) =", p_data_given_H2

    p_H1_given_data = (p_data_given_H1 * p_H1) / (p_data_given_H1 * p_H1 + p_data_given_H2 * p_H2)
    p_H2_given_data = (p_data_given_H2 * p_H2) / (p_data_given_H1 * p_H1 + p_data_given_H2 * p_H2)
    return p_H1_given_data, p_H2_given_data


if __name__ == '__main__':

    np.random.seed(0)

    iterations = 10000

    c = 28
    n = 40
    p_H1 = 0.5
    p_H1_given_data, p_H2_given_data = compute_posteriors(c, n, p_H1=p_H1, iterations=iterations)

    print
    print "p(H1|data) =", p_H1_given_data
    print "p(H2|data) =", p_H2_given_data
    print "BF_21 =", p_H2_given_data / p_H1_given_data * p_H1 / (1.0 - p_H1)
