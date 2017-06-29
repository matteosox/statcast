import numpy as np

from sklearn.neighbors.kde import KernelDensity
from sklearn.utils.validation import check_array

from scipy import stats
from scipy.special import gamma

from .base import BetterModel
from .spark import GridSearchCV


def ballVol(r, n):
    return np.pi ** (n / 2) * r ** n / gamma(n / 2 + 1)


def epanechnikov(X, h):
    d = X.shape[1]
    x2 = ((X / h) ** 2).sum(1, keepdims=True)
    y = (d + 2) * ((1 - x2) * ((1 - x2) > 0)) / (2 * ballVol(1, d))
    return y * h ** -d


def tophat(X, h):
    d = X.shape[1]
    x = np.sqrt(((X / h) ** 2).sum(1, keepdims=True))
    y = (x < 1) / ballVol(1, d)
    return y * h ** -d


def gaussian(X, h):
    d = X.shape[1]
    x2 = ((X / h) ** 2).sum(1, keepdims=True)
    y = np.exp(-x2 / 2) / (2 * np.pi) ** (d / 2)
    return y * h ** -d


def exponential(X, h):
    d = X.shape[1]
    x = np.sqrt(((X / h) ** 2).sum(1, keepdims=True))
    y = np.exp(-x) / ballVol(1, d) / np.math.factorial(d)
    return y * h ** -d


def linear(X, h):
    d = X.shape[1]
    if d > 1:
        raise NotImplementedError('Linear kernel not implemented for dims > 1')
    x = np.sqrt(((X / h) ** 2).sum(1, keepdims=True))
    y = (1 - x) * (x < 1)
    return y * h ** -d


def cosine(X, h):
    d = X.shape[1]
    if d > 1:
        raise NotImplementedError('Cosine kernel not implemented for dims > 1')
    x = np.sqrt(((X / h) ** 2).sum(1, keepdims=True))
    y = np.cos(x * np.pi / 2) * (x < 1) * np.pi / 4
    return y * h ** -d

kernelFunctions = {'gaussian': gaussian,
                   'tophat': tophat,
                   'epanechnikov': epanechnikov,
                   'exponential': exponential,
                   'linear': linear,
                   'cosine': cosine}


class BetterKernelDensity(KernelDensity, BetterModel):
    '''Doc String'''

    def _se(self, X):
        '''Doc String'''

        n = self.trainX_.shape[0]

        Y = np.array([self._kernelFunction(X - row)
                     for row in self.trainX_])
        s2 = Y.var(axis=0, ddof=1)
        return np.sqrt(s2 / n).flatten()

    def predict(self, X):
        ''' Doc String'''

        return np.exp(self.score_samples(X))

    def confidence(self, X, alpha=0.05):
        '''Doc String'''

        X = check_array(X)
        a, b = self.trainX_.min(), self.trainX_.max()
        if self.kernel == 'gaussian':
            w = 3
        elif self.kernel == 'exponential':
            w = 2 * stats.expon.ppf(2 * (stats.norm.cdf(1.5) - 0.5))
        else:
            w = 2
        m = (b - a) / (w * self.bandwidth)
        q = stats.norm.ppf((1 + (1 - alpha) ** (1 / m)) / 2)

        f = self.predict(X)
        se = self._se(X)
        return f - q * se, f + q * se

    def confidenceD(self, data, alpha=0.05):
        '''Doc String'''

        X = self.createX(data)
        return self.confidence(X, alpha)

    def _kernelFunction(self, X):
        '''Doc String'''

        return kernelFunctions[self.kernel](X, self.bandwidth)

    def selectBandwidth(self, bandwidths=None, n_jobs=1, cv=None):
        '''Doc String'''

        if bandwidths is None:
            xmins, xmaxs = self.trainX_.min(0), self.trainX_.max(0)
            bandwidths = np.logspace(-3, -1, num=10) * (xmaxs - xmins).max()

        parameters = {'bandwidth': bandwidths}
        trainGrid = GridSearchCV(self, parameters, cv=cv,
                                 n_jobs=n_jobs, refit=False).fit(self.trainX_)
        self.bandwidth = trainGrid.best_params_['bandwidth']
        self.cv_results_ = trainGrid.cv_results_
        return self
