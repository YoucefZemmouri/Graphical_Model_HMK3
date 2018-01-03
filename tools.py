import numpy as np
from math import *
from numpy.linalg import det, inv


def Gaussian(x, mu, Sigma):
    d = mu.shape[0]
    c = 1 / np.sqrt(((2 * pi) ** d) * det(Sigma))
    temp = ((x - mu).T).dot(inv(Sigma)).dot(x - mu)
    return c * np.exp(-temp / 2)


def get_log_alpha(U, pi, mus, Sigmas, A):
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


def get_log_beta(U, pi, mus, Sigmas, A):
    T = U.shape[0]
    K = len(pi)
    log_beta = np.zeros((T, K))

    log_beta[T - 1, :] = 0

    for t in range(T - 2, -1, -1):
        for k in range(K):
            a = np.zeros(K)
            for i in range(K):
                a[i] = np.log(Gaussian(U[t + 1, :], mus[i], Sigmas[i])) + log_beta[t + 1, i] + np.log(A[i, k])
            a_ = np.max(a)
            log_beta[t, k] = a_ + np.log(np.sum(np.exp(a - a_)))
    return log_beta


def get_log_likelihood(log_alpha, log_beta):
    temp = (log_alpha + log_beta)[0, :]
    return np.log(np.sum(np.exp(temp - np.max(temp)))) + np.max(temp)


def get_log_likelihood_all(log_alpha, log_beta):
    temp = log_alpha + log_beta
    max_ = np.tile(np.max(temp, axis=1), (temp.shape[1], 1)).transpose()
    likelihood_all = np.log(np.sum(np.exp(temp - max_), axis=1)) + np.max(temp, axis=1)
    return likelihood_all


def apply_smoothing(log_alpha, log_beta):
    # Compute p(q_t|u_0,...,u_T)
    num = log_alpha + log_beta
    likelihood_all = get_log_likelihood_all(log_alpha, log_beta)
    denum = np.tile(likelihood_all, (num.shape[1], 1)).transpose()
    return np.exp(num - denum)


def get_prob(log_alpha, log_beta, A, mus, Sigmas, U):
    # proba[t,i,j] = p[q_t = i, q_t+1 = j|u_0,...,u_T]
    T = U.shape[0]
    K = A.shape[0]
    prob = np.zeros((T - 1, K, K))

    likelihood_all = get_log_likelihood_all(log_alpha, log_beta)

    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                prob[t, i, j] = np.exp(log_alpha[t, i] + log_beta[t + 1, j] + np.log(A[j, i]) + np.log(
                    Gaussian(U[t + 1], mus[j], Sigmas[j])) - likelihood_all[t])
    return prob


def EM(U, U_test, A, Pi, mus, Sigmas, epsilon=0.0001):
    T = U.shape[0]
    K = A.shape[0]
    log_likelihood_train = []
    log_likelihood_test = []

    log_likelihood_old = -np.inf

    log_alpha = get_log_alpha(U, Pi, mus, Sigmas, A)
    log_beta = get_log_beta(U, Pi, mus, Sigmas, A)

    log_alpha_test = get_log_alpha(U_test, Pi, mus, Sigmas, A)
    log_beta_test = get_log_beta(U_test, Pi, mus, Sigmas, A)

    log_likelihood = get_log_likelihood(log_alpha, log_beta)
    log_likelihood_train.append(log_likelihood)
    log_likelihood_test.append(get_log_likelihood(log_alpha_test, log_beta_test))

    while np.abs(log_likelihood_old - log_likelihood) > epsilon:
        prob = get_prob(log_alpha, log_beta, A, mus, Sigmas, U)
        smooth = apply_smoothing(log_alpha, log_beta)

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
                temp = U_centered[t].reshape((d, 1))
                Sigma_temp += smooth[t, i] * np.dot(temp, temp.transpose())
            Sigmas[i] = Sigma_temp / np.sum(smooth[:, i])

        log_likelihood_old = log_likelihood

        log_alpha = get_log_alpha(U, Pi, mus, Sigmas, A)
        log_beta = get_log_beta(U, Pi, mus, Sigmas, A)
        log_likelihood = get_log_likelihood(log_alpha, log_beta)
        log_likelihood_train.append(log_likelihood)

        log_alpha_test = get_log_alpha(U_test, Pi, mus, Sigmas, A)
        log_beta_test = get_log_beta(U_test, Pi, mus, Sigmas, A)
        log_likelihood_test.append(get_log_likelihood(log_alpha_test, log_beta_test))

    return Pi, A, mus, Sigmas, log_likelihood_train, log_likelihood_test


def Viterbi(U, mus, Sigmas, A, Pi):
    T = U.shape[0]
    K = A.shape[0]

    X = np.zeros(T, dtype=int)
    Z = np.zeros(T, dtype=int)

    T_1 = np.zeros((K, T))
    T_2 = np.zeros((K, T))

    for k in range(K):
        T_1[k, 0] = np.log(Pi[k]) + np.log(Gaussian(U[0, :], mus[k], Sigmas[k]))
        T_2[k, 0] = - np.inf

    for t in range(1, T):
        for k in range(K):
            T_1[k, t] = np.max(T_1[:, t - 1] + np.log(A[:, k])) + np.log(Gaussian(U[t, :], mus[k], Sigmas[k]))
            T_2[k, t] = np.argmax(T_1[:, t - 1] + np.log(A[:, k]))

    Z[T - 1] = np.argmax(T_1[:, T - 1])
    X[T - 1] = Z[T - 1]

    for t in range(T - 2, -1, -1):
        Z[t] = T_2[Z[t + 1], t + 1]
        X[t] = Z[t]
    return X



