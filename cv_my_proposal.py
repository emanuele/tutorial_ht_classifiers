import numpy as np
from example_jeffreys import compute_posteriors

if __name__ == '__main__':

    np.random.seed(0)
    cs = np.array([14, 9, 17, 16, 13])
    n = 20
    p_H1 = 0.5
    iterations = 10000
    p_H1_given_data = []
    p_H2_given_data = []
    for c in cs:
        post_H1, post_H2 = compute_posteriors(c, n, p_H1=p_H1, iterations=iterations, verbose=False)
        print post_H1, post_H2, post_H2 / post_H1
        p_H1_given_data.append(post_H1)
        p_H2_given_data.append(post_H2)

    mean_p_H1_given_data = np.mean(p_H1_given_data)
    mean_p_H2_given_data = np.mean(p_H2_given_data)
    print
    print mean_p_H1_given_data, mean_p_H2_given_data, mean_p_H1_given_data + mean_p_H2_given_data, mean_p_H2_given_data / mean_p_H1_given_data
