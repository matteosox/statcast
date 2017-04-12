import numpy as np

from sklearn.neighbors.kde import KernelDensity
from sklearn.utils.validation import check_array

from scipy import stats

from .base import BetterModel


class BetterKernelDensity(KernelDensity, BetterModel):
    '''Doc String'''

    def _kernelFunction(self, X):
        '''Doc String'''

        if not hasattr(self, '_kde'):
            self._kde = \
                self.__class__(**self.get_params()).fit(np.zeros((1, 1)))
        elif not self.get_params() == self._kde.get_params():
            self._kde.set_params(**self.get_params())
        elif not X.shape[1] == self._kde.tree_.data.shape[1]:
            self._kde.fit(np.zeros((1, X.shape[1])))
        return np.exp(self._kde.score_samples(X))

    def _se(self, X):
        '''Doc String'''

        trainData = self.tree_.data
        n = self.tree_.data.shape[0]

        Y = np.array([self._kernelFunction(X - row)
                     for row in trainData])
        s2 = Y.var(axis=0, ddof=1)
        return np.sqrt(s2 / n)

    def predict(self, X):
        ''' Doc String'''

        return np.exp(self.score_samples(X))

    def confidence(self, X, alpha=0.05):
        '''Doc String'''

        X = check_array(X)
        trainData = np.array(self.tree_.data)
        a, b = trainData.min(), trainData.max()
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
