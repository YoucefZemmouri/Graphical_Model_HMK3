import numpy as np
from math import *
from numpy.linalg import det, inv


def Gaussian(x, mu, Sigma):
    d = mu.shape[0]
    c = 1 / np.sqrt(((2 * pi) ** d) * det(Sigma))
    temp = ((x - mu).T).dot(inv(Sigma)).dot(x - mu)
    return c * np.exp(-temp / 2)


def log_alpha(U, pi, mus, Sigmas, A):
    T = U.shape[0]
    K = len(pi)
    log_alpha = np.zeros((T, K))

    for k in range(K):
        log_alpha[0, k] = np.log(pi[k]) + np.log(Gaussian(U[0, :], mus[k], Sigmas[k]))

    for t in range(1, T):
        for k in range(K):
            a = log_alpha[t - 1, :] + np.log(A[k, :])
            a_ = np.max(a)
            log_alpha[t, k] = np.log(Gaussian(U[t, :], mus[k], Sigmas[k])) + a_ + np.log(np.sum(np.exp(a - a_)))

    return log_alpha


def log_beta(U, pi, mus, Sigmas, A):
    T = U.shape[0]
    K = len(pi)
    log_beta = np.zeros((T, K))

    log_beta[T - 1, :] = 1

    for t in range(T - 2, -1, -1):
        for k in range(K):
            a = np.zeros(K)
            for i in range(K):
                a[i] = np.log(Gaussian(U[t, :], mus[i], Sigmas[i])) + log_beta[t + 1, i] + np.log(A[i, k])
            a_ = np.max(a)
            log_beta[t, k] = a_ + log(np.sum(np.exp(a - a_)))
    return log_beta


def log_likelihood(log_alpha, log_beta):
    temp1 = log_alpha + log_beta
    max_1 = np.max(temp1, axis=1)
    max_ = np.tile(max_1, (temp1.shape[1], 1)).transpose()
    temp2 = np.log(np.sum(np.exp(temp1 - max_), axis=1)) + max_1
    return temp2.mean()

'''
def log_likelihood_test(log_alpha, log_beta):
    temp = (log_alpha + log_beta)[3,:]
    max = np.max(temp)
    return np.sum(np.exp(temp - max)) + max
'''

def smoothing(log_alpha, log_beta):
    # Compute p(q_t|u_0,...,u_T)
    temp1 = log_alpha + log_beta
    max_1 = np.max(temp1, axis=1)
    max_ = np.tile(max_1, (temp1.shape[1], 1)).transpose()
    temp2 = np.log(np.sum(np.exp(temp1 - max_), axis=1)) + max_1
    temp2 = np.tile(temp2, (temp1.shape[1], 1)).transpose()
    return np.exp(temp1 - temp2)


def proba(log_alpha, log_beta, A, mus, Sigmas, U):
    # proba[t,i,j] = p[q_t = i, q_t+1 = j|u_0,...,u_T]
    T = U.shape[0]
    K = A.shape[0]
    proba = np.zeros((T - 1, K, K))

    temp1 = (log_alpha + log_beta)[:T - 1, :]
    max_1 = np.max(temp1, axis=1)
    max_ = np.tile(max_1, (temp1.shape[1], 1)).transpose()
    temp2 = np.log(np.sum(np.exp(temp1 - max_), axis=1)) + max_1
    denom = np.tile(temp2, (temp1.shape[1], 1)).transpose()

    for j in range(K):
        table = np.zeros((T - 1, K))
        for i in range(T - 1):
            table[i] = np.log(Gaussian(U[i], mus[j], Sigmas[j]))
        A_log = np.tile(np.log(A[j, :]), (T - 1, 1))
        log_q_t1 = np.tile(log_beta[1:, j], (K, 1)).transpose()
        proba[:, :, j] = np.exp(log_alpha[:T - 1, :] + log_q_t1 + A_log + table - denom)
    return proba


def EM(U, U_test, A, Pi, mus, Sigmas, epsilon=0.0001):
    T = U.shape[0]
    K = A.shape[0]
    log_likelihood_train = []
    log_likelihood_test = []

    log_likelihood_old = -np.inf
    log_likelihood = 0

    while np.abs(log_likelihood_old-log_likelihood) > epsilon:
        logalpha = log_alpha(U, Pi, mus, Sigmas, A)
        logbeta = log_beta(U, Pi, mus, Sigmas, A)
        prob = proba(logalpha, logbeta, A, mus, Sigmas, U)
        smooth = smoothing(logalpha, logbeta)

        Pi = smooth[0, :]
        for i in range(K):
            for j in range(K):
                A[i, j] = np.sum(prob[:, j, i]) / np.sum(smooth[:, j])
            mus[i] = np.sum(np.multiply(U, np.tile(smooth[:, i], (U.shape[1], 1)).transpose()), axis=0) / np.sum(
                smooth[:, i])
            U_centered = U - mus[i]
            d = U.shape[1]
            Sigma_temp = np.zeros((d, d))
            for t in range(T):
                temp = U_centered[t].reshape((d,1))
                Sigma_temp += smooth[t, i] * np.dot(temp, temp.transpose())
            Sigmas[i] = Sigma_temp / np.sum(smooth[:, i])
        log_likelihood_old = log_likelihood

        logalpha = log_alpha(U, Pi, mus, Sigmas, A)
        logbeta = log_beta(U, Pi, mus, Sigmas, A)

        temp1 = logalpha + logbeta
        max_1 = np.max(temp1, axis=1)
        max_ = np.tile(max_1, (temp1.shape[1], 1)).transpose()
        temp2 = np.log(np.sum(np.exp(temp1 - max_), axis=1)) + max_1
        log_likelihood = temp2.mean()

        log_alpha_test = log_alpha(U_test, Pi, mus, Sigmas, A)
        log_beta_test = log_beta(U_test, Pi, mus, Sigmas, A)

        #log_likelihood = log_likelihood(logalpha, logbeta)

        log_likelihood_train.append(log_likelihood)

        temp1 = log_alpha_test + log_beta_test
        max_1 = np.max(temp1, axis=1)
        max_ = np.tile(max_1, (temp1.shape[1], 1)).transpose()
        temp2 = np.log(np.sum(np.exp(temp1 - max_), axis=1)) + max_1
        log_likelihood_tst = temp2.mean()
        log_likelihood_test.append(log_likelihood_tst)

    return Pi, A, mus, Sigmas, log_likelihood_train, log_likelihood_test

