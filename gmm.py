import numpy as np
from scipy.stats import multivariate_normal


class _Cluster:
    def __init__(self, d):
        self.d = d
        self.mean = np.random.uniform(0, 1, d)
        self.var = np.eye(d)
        self.prior = 1

    def prob(self, x):
        return multivariate_normal.pdf(x, self.mean, self.var)

    def update(self, mean, var, prior):
        self.mean = mean
        self.var = var
        self.prior = prior


class GMM:
    def __init__(self, k, tolerance=1e-3, reg_cov=1e-6):
        self.tol = tolerance
        self.reg_cov = reg_cov
        self.k = k
        self.d = None
        self.clusters = None
        self.thresh = 0
        self.avg_prob = 0

    def fit(self, data):
        self.d = data.shape[1]
        self._initilize_clusters(self.d, self.k)

        while True:
            avg_prob = self.avg_prob

            b = self._e_step(data)

            self._m_step(data, b)

            if abs(self.avg_prob - avg_prob) < self.tol:
                break

    def prob(self, data) -> np.ndarray:
        """ calculate the likelihood of new data """
        n, d = data.shape

        # p(x|k)p(k)
        p = np.empty((n, self.k), dtype=data.dtype)
        for i, cluster in enumerate(self.clusters):
            p[:, i] = cluster.prob(data) * cluster.prior

        p = np.sum(p, axis=1)

        return p

    def _initilize_clusters(self, d, k):
        self.clusters = [_Cluster(d) for _ in range(k)]

    def _e_step(self, data: np.ndarray):
        n, d = data.shape

        # p(x|k)
        p_x_k = np.empty((n, self.k), dtype=data.dtype)
        for i, cluster in enumerate(self.clusters):
            p_x_k[:, i] = cluster.prob(data)

        # p(k|x)
        p_k_x = p_x_k  # since we do not use p(x|k) anymore, we will override it
        for i, cluster in enumerate(self.clusters):
            p_k_x[:, i] *= cluster.prior  # p(x|k) * p(k)

        # normalize
        p_x = np.sum(p_k_x, axis=1, keepdims=True)  # p(x) = sum p(x|k)p(k)
        p_x += 10 * np.finfo(data.dtype).eps  # for the 0/0 cases
        p_k_x /= p_x

        self.thresh = np.min(p_x)
        self.avg_prob = np.mean(p_x)

        return p_k_x  # p(k|x)

    def _m_step(self, data, b):
        for k, cluster in enumerate(self.clusters):
            bk = b[:, k:k+1]
            sum_bk = np.sum(bk)

            # calculate mean
            mu = np.mean(data, axis=0)

            # calculate covariance matrix
            m = data - mu
            m1 = np.expand_dims(m, axis=1)
            m2 = np.expand_dims(m, axis=2)
            sigma = np.sum(np.expand_dims(bk, axis=2) * (m1 * m2), axis=0) / sum_bk
            sigma.flat[::self.d + 1] += self.reg_cov

            prior = sum_bk / data.shape[0]

            cluster.update(mu, sigma, prior)
