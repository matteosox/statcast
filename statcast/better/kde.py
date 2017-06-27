import numpy as np

from sklearn.neighbors.kde import KernelDensity
from sklearn.utils.validation import check_array

from scipy import stats

from .base import BetterModel
from .spark import GridSearchCV


def gaussian(X, h):
    return stats.norm.pdf(X / h).prod(1) / h


def tophat(X, h):
    return ((np.abs(X / h) < 1) / 2).prod(1) / h


def epanechnikov(X, h):
    return ((np.abs(X / h) < 1) * 3 / 4 * (1 - (X / h) ** 2)).prod(1) / h


def exponential(X, h):
    return (np.exp(-np.abs(X / h)) / 2).prod(1) / h


def linear(X, h):
    return ((np.abs(X / h) < 1) * (1 - np.abs(X / h))).prod(1) / h


def cosine(X, h):
    return ((np.abs(X / h) < 1) *
            np.cos(X / h * np.pi / 2)).prod(1) / h * np.pi / 4

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
        return np.sqrt(s2 / n)

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
        self.bandwidth = trainGrid.best_estimator_.bandwidth
        self.cv_results_ = trainGrid.cv_results_
        return self
