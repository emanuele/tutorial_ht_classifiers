"""This experiment compares the effects of the statistical dependency
between the predictions across the folds of cross-validation when
performing statistical tests. The comparison is computed over:
- Student's 1-sample t-test on the accuracies across the folds.
- Binomial test on the sum of the correct predictions across the folds.
- Bayesian Beta-Binomial test (see Olivetti 2012 Pat.Rec.) on the sum
  of the correct predictions across the folds.

In order to study the statistical dependency though simulation, we
compare the number of correct predictions across the folds of
cross-validation against the number of correct predictions of fresh
new dataset (so 'independent') sampled from the same distribution and
of the same size used in the CV folds.

We investigate two strategies for generating 'indpendent' measure of
correct predictions across folds:
1) We generate a new dataset at each fold and split it in train and
test parts in the same way we do for the cross-validation folds.

2) We generate just one new dataset for all folds and use it all as
test set of one (at random, but in this implementation we use the last
one to keep the code simple) of the classifiers trained during the
CV-folds.

The second strategy cannot be used with the t-test, so for that case
we use just the first strategy.
"""

from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from scipy.stats import binom as binomial_distribution, t as student_t
from example_jeffreys import compute_posteriors
from scipy.special import gammaln


def beta_binomial_gammaln(k, n, a, b):
    """The pmf of the Beta-binomial distribution. Computation based
    on gammaln function.
    
    Note: this implementation suffers much less the numerical issues
    of beta_binomial().
    
    See: http://en.wikipedia.org/wiki/Beta-binomial_distribution
    
    k = a vector of non-negative integers <= n
    n = an integer
    a = an array of non-negative real numbers
    b = an array of non-negative real numbers
    """
    tmp0 = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))
    tmp1 = gammaln(a+k) + gammaln(n+b-k)
    tmp2 = - gammaln(a+b+n) + gammaln(a+b) - (gammaln(a) + gammaln(b))
    # return np.exp((tmp0 + tmp1 + tmp2).sum(0))
    return np.exp(tmp0 + tmp1 + tmp2)


def generate_data(size=100, n_features=2, std=None):
    if std is None: std = np.sqrt(n_features / 3.0)
    size0 = int(size / 2)
    size1 = int(size - size0) # casting to int is necessary to prevent a numpy bug about int64 size argument below
    y = np.array([0] * size0 + [1] * size1, dtype=np.int)
    mu0 = np.zeros(n_features)
    cov = np.eye(n_features) * std
    X0 = np.random.multivariate_normal(mean=mu0, cov=cov, size=size0)
    mu1 = np.ones(n_features)
    X1 = np.random.multivariate_normal(mean=mu1, cov=cov, size=size1)
    X = np.vstack([X0, X1])
    return X, y


if __name__ == '__main__':

    np.random.seed(1)
    
    cv = 5
    d = 50
    std = 5.0
    N = 100
    n_sizes = 20
    iterations = 10000
    classifier = LogisticRegression # SVC # KNeighborsClassifier # 
    classifier_parameters = {}
    strategy = 1
    save = True
    
    sizes = np.round(np.linspace(cv*2, N, n_sizes)).astype(np.int)
    
    acc_cv_mean = np.zeros(len(sizes))
    acc_cv_var = np.zeros(len(sizes))
    acc_cv_t = np.zeros(len(sizes))
    acc_cv_p_value = np.zeros(len(sizes))
    correct_cv_sum = np.zeros(len(sizes))
    correct_cv_bintest = np.zeros(len(sizes))
    acc_indep_mean = np.zeros(len(sizes))
    acc_indep_var = np.zeros(len(sizes))
    acc_indep_t = np.zeros(len(sizes))
    acc_indep_p_value = np.zeros(len(sizes))
    correct_indep_sum = np.zeros(len(sizes))
    correct_indep_bintest = np.zeros(len(sizes))
    correct_cv_posteriorH2 = np.zeros(len(sizes))
    correct_indep_posteriorH2 = np.zeros(len(sizes))
    correct_cv_avg_posteriorH2 = np.zeros(len(sizes))
    correct_indep_avg_posteriorH2 = np.zeros(len(sizes))    
    
    for j, size in enumerate(sizes):
        print "size:", size

        expected = np.zeros((iterations,cv))
        pe = np.zeros(iterations)

        testset_size = np.zeros((iterations,cv), dtype=np.float)
        correct_cv = np.zeros((iterations,cv), dtype=np.float)
        correct_indep = np.zeros((iterations,cv), dtype=np.float)
        correct_indep2 = np.zeros(iterations, dtype=np.float)
        for i in range(iterations):
            X_cv, y_cv = generate_data(size, n_features=d, std=std)
            skf = StratifiedKFold(y_cv, n_folds=cv)
            for k, (train, test) in enumerate(skf):
                testset_size[i,k] = test.size
                # CV:
                clf_cv = classifier(**classifier_parameters)
                clf_cv.fit(X_cv[train], y_cv[train])
                y_pred_cv = clf_cv.predict(X_cv[test])
                correct_cv[i,k] = (y_pred_cv == y_cv[test]).sum()

                # For the "independent" case, strategy 1, we draw a new dataset for each fold:
                X_indep, y_indep = generate_data(size, n_features=d, std=std) # we assume y_indep==y_cv !
                clf_indep = classifier(**classifier_parameters)
                clf_indep.fit(X_indep[train], y_indep[train])
                y_pred_indep = clf_indep.predict(X_indep[test])
                correct_indep[i,k] = (y_pred_indep == y_indep[test]).sum()

            # For the "independent" case, strategy 2, we draw an entire new dataset just for test:
            X_indep2, y_indep2 = generate_data(size, n_features=d, std=std)
            y_pred_indep2 = clf_indep.predict(X_indep2) # We use the last indep (or cv) classifier
            correct_indep2[i] = (y_pred_indep2 == y_indep2).sum()
            

        accuracies_cv = correct_cv / testset_size
        acc_cv_mean[j] = accuracies_cv.mean(1).mean()
        acc_cv_var[j] = accuracies_cv.var(axis=1, ddof=1).mean()
        print "CV mean, var, std:", acc_cv_mean[j], acc_cv_var[j], np.sqrt(acc_cv_var[j])

        accuracies_indep = correct_indep / testset_size
        acc_indep_mean[j] = accuracies_indep.mean(1).mean()
        acc_indep_var[j] = accuracies_indep.var(axis=1, ddof=1).mean()
        print "Independent mean, var, std:", acc_indep_mean[j], acc_indep_var[j], np.sqrt(acc_indep_var[j])

        # Student's 1-sample t-test over the accuracies of the folds.
        acc_cv_t[j] = (acc_cv_mean[j] - 0.5) /  np.sqrt(acc_cv_var[j] / cv)
        acc_cv_p_value[j] = 1.0 - student_t.cdf(acc_cv_t[j], cv-1)
        print "CV t, p-value:", acc_cv_t[j], acc_cv_p_value[j]

        acc_indep_t[j] = (acc_indep_mean[j] - 0.5) / np.sqrt(acc_indep_var[j] / cv)
        acc_indep_p_value[j] = 1.0 - student_t.cdf(acc_indep_t[j], cv-1)
        print "Independent t, p-value:", acc_indep_t[j], acc_indep_p_value[j]

        # Summing the correct predictions over the fold and computing a Binomial test
        if strategy == 1:
            correct_indep_sum1 = correct_indep.sum(1) # This refers to strategy 1
        elif strategy == 2:
            correct_indep_sum1 = correct_indep2 # This refers to strategy 2
        else:
            raise Exception
        
        correct_cv_sum[j] = correct_cv.sum(1).mean()
        correct_indep_sum[j] = correct_indep_sum1.mean()
        print "correct sum CV, indep:", correct_cv_sum[j], correct_indep_sum[j]

        correct_cv_bintest[j] = (1.0 - binomial_distribution.cdf(correct_cv.sum(1) - 1, size, 0.5)).mean()
        correct_indep_bintest[j] = (1.0 - binomial_distribution.cdf(correct_indep_sum1 - 1, size, 0.5)).mean()
        print "Bintest pvalue CV, indep:", correct_cv_bintest[j], correct_indep_bintest[j]

        # Summing the correct predictions over the folds and computing p(H2|sum) (Bayesian Binomial)
        p_H1 = 0.5
        p_H2 = 0.5
        p_data_given_H1_cv = binomial_distribution.pmf(correct_cv.sum(1), size, 0.5)
        p_data_given_H2_cv = beta_binomial_gammaln(correct_cv.sum(1), size, 1.0, 1.0)
        correct_cv_posteriorH2[j] = ((p_data_given_H2_cv * p_H2) / (p_data_given_H1_cv * p_H1 + p_data_given_H2_cv * p_H2)).mean()
        p_data_given_H1_indep = binomial_distribution.pmf(correct_indep_sum1, size, 0.5)
        p_data_given_H2_indep = beta_binomial_gammaln(correct_indep_sum1, size, 1.0, 1.0)
        correct_indep_posteriorH2[j] = ((p_data_given_H2_indep * p_H2) / (p_data_given_H1_indep * p_H1 + p_data_given_H2_indep * p_H2)).mean()
        print "p(H2|data) CV, indep:", correct_cv_posteriorH2[j], correct_indep_posteriorH2[j]


        # Averaging the posteriors of H2 over folds:
        p_H1 = 0.5
        p_H2 = 0.5
        p_data_given_H1_cv = binomial_distribution.pmf(correct_cv, testset_size, 0.5)
        p_data_given_H2_cv = beta_binomial_gammaln(correct_cv, testset_size, 1.0, 1.0)
        correct_cv_avg_posteriorH2[j] = ((p_data_given_H2_cv * p_H2) / (p_data_given_H1_cv * p_H1 + p_data_given_H2_cv * p_H2)).mean(1).mean()

        p_data_given_H1_indep = binomial_distribution.pmf(correct_indep, testset_size, 0.5)
        p_data_given_H2_indep = beta_binomial_gammaln(correct_indep, testset_size, 1.0, 1.0)
        correct_indep_avg_posteriorH2[j] = ((p_data_given_H2_indep * p_H2) / (p_data_given_H1_indep * p_H1 + p_data_given_H2_indep * p_H2)).mean(1).mean()
        print "AVG. p(H2|data) CV, indep:", correct_cv_avg_posteriorH2[j], correct_indep_avg_posteriorH2[j]


    # mean
    plt.figure()
    plt.plot(sizes, acc_indep_mean, 'k--', label='expected')
    plt.plot(sizes, acc_cv_mean, 'k-', label='%d-CV' % cv)
    plt.xlabel('dataset size ($N$) ')
    plt.ylabel('mean(accuracy)')
    plt.title('Mean Accuracy (k=%d, iterations=%d)' % (cv,iterations))
    plt.legend(loc='lower right')
    if save:  plt.savefig('cv%d_d%d_mean_accuracy.pdf' % (cv, d))

    # var
    plt.figure()
    plt.plot(sizes, acc_indep_var, 'k--', label='expected')
    plt.plot(sizes, acc_cv_var, 'k-', label='%d-CV' % cv)
    plt.ylim(ymin=0.0)
    plt.xlabel('dataset size ($N$) ')
    plt.ylabel('variance(accuracy)')
    plt.title('Variance of the Accuracy (k=%d, iterations=%d)' % (cv,iterations))
    plt.legend()
    if save: plt.savefig('cv%d_d%d_variance_accuracy.pdf' % (cv, d))

    # std
    acc_indep_std = np.sqrt(acc_indep_var)    
    acc_cv_std = np.sqrt(acc_cv_var)
    plt.figure()
    plt.plot(sizes, acc_indep_std, 'k--', label='expected')
    plt.plot(sizes, acc_cv_std, 'k-', label='%d-CV' % cv)
    plt.xlabel('dataset size ($N$) ')
    plt.ylabel('std(accuracy)')
    plt.title('Standard Deviation of the Accuracy (k=%d, iterations=%d)' % (cv,iterations))
    plt.legend()
    if save: plt.savefig('cv%d_d%d_std_accuracy.pdf' % (cv, d))

    # t
    plt.figure()
    plt.plot(sizes, acc_indep_t, 'k--', label='expected')
    plt.plot(sizes, acc_cv_t, 'k-', label='%d-CV' % cv)
    plt.xlabel('dataset size ($N$) ')
    plt.ylabel('$t$')
    plt.title('Across-folds $t$ of the Accuracies (k=%d, iterations=%d)' % (cv,iterations))
    plt.legend()
    if save: plt.savefig('cv%d_d%d_std_t.pdf' % (cv, d))

    # p-value of t-test
    acc_cv_tt_p_value = 1.0 - student_t.cdf(acc_cv_t, cv-1)
    acc_indep_tt_p_value = 1.0 - student_t.cdf(acc_indep_t, cv-1)
    plt.figure()
    plt.plot(sizes, acc_indep_tt_p_value, 'k--', label='expected')
    plt.plot(sizes, acc_cv_tt_p_value, 'k-', label='%d-CV' % cv)
    plt.xlabel('dataset size ($N$) ')
    plt.ylabel('$t$')
    plt.title('Across-folds $p$-value of the $t$-test of the Accuracies (k=%d, iterations=%d)' % (cv,iterations))
    plt.legend()
    if save: plt.savefig('cv%d_d%d_ttest_pvalue.pdf' % (cv, d))

    # CV decrease of std
    plt.figure()
    plt.plot(sizes, (acc_indep_std - acc_cv_std) / acc_indep_std, 'k-')
    plt.xlabel('dataset size ($N$) ')
    plt.title('CV decrease of std')

    # CV decrease of the t value
    plt.figure()
    plt.plot(sizes, (acc_indep_t - acc_cv_t) / acc_indep_t, 'k-')
    plt.xlabel('dataset size ($N$) ')
    plt.title('CV decrease of t')

    # CV decrease of the p-value
    plt.figure()
    plt.plot(sizes, (acc_indep_p_value - acc_cv_p_value) / acc_indep_p_value, 'k-')
    plt.xlabel('dataset size ($N$) ')
    plt.title('CV decrease of p-value (k=%d, iterations=%d)' % (cv,iterations))
    if save: plt.savefig('cv%d_d%d_ttest_decrease_pvalue.pdf' % (cv, d))

    # Sum of correct predictions
    plt.figure()
    plt.plot(sizes, correct_indep_sum, 'k--', label='indep')
    plt.plot(sizes, correct_cv_sum, 'k-', label='CV')
    plt.xlabel('dataset size ($N$) ')
    plt.title('Sum of correct predictions')
    plt.legend()

    # p-value of the Binomial test of the sum of correct predictions
    plt.figure()
    plt.plot(sizes, correct_indep_bintest, 'k--', label='indep')
    plt.plot(sizes, correct_cv_bintest, 'k-', label='CV')
    plt.title('p-value of the Bin.test of the sum of correct pred. (k=%d, iterations=%d)' % (cv,iterations))
    plt.xlabel('dataset size ($N$) ')
    plt.legend()
    if save: plt.savefig('cv%d_d%d_binom_pvalue.pdf' % (cv, d))

    # Increase of p-value of CV
    plt.figure()
    plt.plot(sizes, (correct_cv_bintest - correct_indep_bintest) / correct_cv_bintest, 'k-')
    plt.xlabel('dataset size ($N$) ')
    plt.title('CV increase of p-value (k=%d, iterations=%d)' % (cv,iterations))
    if save: plt.savefig('cv%d_d%d_binom_increase_pvalue.pdf' % (cv, d))

    # p(H2|data) of the sum of correct predictions
    plt.figure()
    plt.plot(sizes, correct_indep_posteriorH2, 'k--', label='indep')
    plt.plot(sizes, correct_cv_posteriorH2, 'k-', label='CV')
    plt.title(r'$p(H_2|$data$)$ of the sum of correct predictions (k=%d, iterations=%d)' % (cv,iterations))
    plt.xlabel('dataset size ($N$) ')
    plt.legend()
    if save: plt.savefig('cv%d_d%d_posteriorH2.pdf' % (cv, d))

    plt.figure()
    plt.plot(sizes, (correct_cv_posteriorH2 - correct_indep_posteriorH2) / correct_cv_posteriorH2, 'k-')
    plt.xlabel('dataset size ($N$) ')
    plt.title('CV relative variation in Posterior H2 (k=%d, iterations=%d)' % (cv,iterations))
    if save: plt.savefig('cv%d_d%d_variation_posteriorH2.pdf' % (cv, d))

    # AVG p(H2|data) of the sum of correct predictions
    plt.figure()
    plt.plot(sizes, correct_indep_avg_posteriorH2, 'k--', label='indep')
    plt.plot(sizes, correct_cv_avg_posteriorH2, 'k-', label='CV')
    plt.title(r'AVG $p(H_2|$data$)$ of the sum of correct predictions (k=%d, iterations=%d)' % (cv,iterations))
    plt.xlabel('dataset size ($N$) ')
    plt.legend()
    if save: plt.savefig('cv%d_d%d_avg_posteriorH2.pdf' % (cv, d))

    plt.figure()
    plt.plot(sizes, (correct_cv_avg_posteriorH2 - correct_indep_avg_posteriorH2) / correct_cv_avg_posteriorH2, 'k-')
    plt.xlabel('dataset size ($N$) ')
    plt.title('AVG. CV relative variation in Posterior H2 (k=%d, iterations=%d)' % (cv,iterations))
    if save: plt.savefig('cv%d_d%d_variation_avg_posteriorH2.pdf' % (cv, d))

    if d == 2:
        plt.figure()
        plt.plot(X_cv[y_cv==0,0], X_cv[y_cv==0,1], 'bo', markersize=14)
        plt.plot(X_cv[y_cv==1,0], X_cv[y_cv==1,1], 'rx', markersize=14, markeredgewidth=4)
        if save: plt.savefig('data_distribution_2D.pdf')
