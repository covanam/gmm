import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, k):
        self.k = k  # number of clusters

        # clusters' parameters:
        self.mu = None  # mean
        self.cov = None  # covarience
        self.pk = None  # possibilities of a cluster

        # thresh to determine whether new data belong to our dataset
        self.thresh = 0

    def fit(self, data: np.ndarray, tol=1e-5, reg=1e-3, max_iter=100):
        """
        :param data: training data of size (N x D)
        :param tol: if the decease of mean likelihood of data after a iteration smaller
                    than tol, the algorithm stops
        :param reg: value add to diaganol of the covarience matrices to ensure them to
                    be non-singular
        :param max_iter: maximum number of iterations
        """
        self._init_clusters(data)

        num_iter = 0
        old_avg_prob = 0
        while True:
            num_iter += 1

            b, avg_prob = self._e_step(data)

            self._m_step(data, b, reg)
            if abs(avg_prob - old_avg_prob) < tol:
                break
            if num_iter == max_iter:
                break

            old_avg_prob = avg_prob

    def _init_clusters(self, data):
        # randomly initilize clusters' parameters
        d = data.shape[1]
        self.mu = np.random.random((self.k, d))
        self.cov = np.zeros((self.k, d, d))
        self.cov[:] = np.eye(d).reshape((1, d, d))
        self.pk = np.ones((1, self.k))

    def prob(self, data) -> np.ndarray:
        """ calculate the likelihood of new data """
        n, d = data.shape

        # p(x|k)p(k)
        p = np.empty((n, self.k), dtype=data.dtype)
        for k in range(self.k):
            p[:, k] = multivariate_normal.pdf(data, self.mu[k], self.cov[k])
        p *= self.pk

        p = np.sum(p, axis=1)

        return p

    def _e_step(self, data: np.ndarray):
        """
        :return:
            - p(k|x) of size (N x K)
            - mean value of data's likelihood p(x)
        """
        n, d = data.shape

        # p(x|k)
        p_x_k = np.empty((n, self.k), dtype=data.dtype)
        for k in range(self.k):
            p_x_k[:, k] = multivariate_normal.pdf(data, self.mu[k], self.cov[k])

        # calculate p(k|x)
        temp = p_x_k * self.pk  # temp = p(x|k) * p(k)
        temp = np.maximum(temp, np.finfo(data.dtype).eps)

        # calculate p(x)
        p_x = np.sum(temp, axis=1, keepdims=True)  # p(x) = sum p(x|k)p(k)

        p_k_x = temp / p_x

        self.thresh = np.min(p_x)
        avg_prob = np.mean(p_x)

        return p_k_x, avg_prob

    def _m_step(self, data, p_k_x, reg=1e-6):
        n, d = data.shape

        sum_pkx = np.sum(p_k_x, axis=0)
        self.mu = p_k_x.T.dot(data) / sum_pkx.reshape((-1, 1))

        for k in range(self.k):
            b = p_k_x[:, k:k + 1]

            # calculate covariance matrix
            m = data - self.mu[k:k + 1]
            m1 = np.expand_dims(m, axis=1)
            m2 = np.expand_dims(m, axis=2)
            self.cov[k] = np.sum(np.expand_dims(b, axis=2) * (m1 * m2), axis=0) / sum_pkx[k]
            self.cov[k].flat[::d + 1] += reg

        self.pk = sum_pkx.reshape((1, self.k)) / data.shape[0]
