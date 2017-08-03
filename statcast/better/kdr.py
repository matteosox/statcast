from inspect import signature

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors.kde import KernelDensity
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.metrics import mean_squared_error

from scipy import stats

from .base import BetterModel
from .kde import BetterKernelDensity


class BetterKDR(BaseEstimator, RegressorMixin, BetterModel):
    '''Doc String'''

    _params = [param for param in signature(KernelDensity).parameters.values()]

    def __init__(self):
        '''Doc String'''

        self.kde = BetterKernelDensity(**self.get_params())

    def fit(self, X, Y):
        '''Doc String'''

        self._flowParams()
        check_X_y(X, Y, multi_output=True, dtype=None)
        self.kde.fit(X)
        self.trainY_ = Y.copy()
        return self

    def _flowParams(self, up=False):
        '''Doc String'''

        if up:
            self.set_params(**self.kde.get_params())
        else:
            self.kde.set_params(**self.get_params())

    def _weights(self, X):
        '''Doc String'''

        self._flowParams()
        trainX = np.array(self.kde.tree_.data)
        num = np.hstack([self.kde._kernelFunction(np.tile(
            row, (trainX.shape[0], 1)) - trainX) for row in X]).T
        den = np.tile(self.kde.predict(X)[:, None] *
                      trainX.shape[0], (1, trainX.shape[0]))
        den[den == 0] = 1
        W = num / den
        W[(W == 0).all(1), :] = 1 / trainX.shape[0]
        return W

    def predict(self, X):
        '''Doc String'''

        check_is_fitted(self, ['trainY_'])
        check_array(X, dtype=None)
        W = self._weights(X)
        return W.dot(self.trainY_)

    def score(self, X, Y, sample_weight=None):
        '''Doc String'''

        X, Y = check_X_y(X, Y, multi_output=True, dtype=None)
        Yp = self.predict(X)
        return -np.sqrt([mean_squared_error(y, yp, sample_weight)
                         for y, yp in zip(Y.T, Yp.T)]).mean()

    def risk(self):
        '''Doc String'''

        check_is_fitted(self, ['trainY_'])
        self._flowParams()
        trainX = np.array(self.kde.tree_.data)
        k0 = self.kde._kernelFunction(np.zeros((1, trainX.shape[1])))
        den = np.tile(1 - k0 / (self.kde.predict(trainX)[:, None] *
                                trainX.shape[0]),
                      (1, self.trainY_.shape[1]))
        num = self.trainY_ - self.predict(trainX)
        risk = ((num / den) ** 2).mean()
        if np.isnan(risk):
            risk = np.inf
        return risk

    def confidence(self, X, alpha=0.05):
        '''Doc String'''

        check_is_fitted(self, ['trainY_'])
        trainX = np.array(self.kde.tree_.data)
        X = check_array(X)
        a, b = trainX.min(0), trainX.max(0)
        if self.kernel == 'gaussian':
            w = 6
        elif self.kernel == 'exponential':
            w = 2 * stats.expon.ppf(2 * (stats.norm.cdf(3) - 0.5))
        else:
            w = 2
        m = np.prod(b - a) / (w * self.bandwidth) ** X.shape[1]
        if self.normalize:
            m *= self.kde.detH_
        q = stats.norm.ppf((1 + (1 - alpha) ** (1 / m)) / 2)

        f = self.predict(X)
        se = np.sqrt((self._weights(X) ** 2).sum(1, keepdims=True) *
                     self.risk())
        return f - q * se, f + q * se

    def confidenceD(self, data, alpha=0.05):
        '''Doc String'''

        X = self.createX(data)
        return self.confidence(X, alpha)

    def selectBandwidth(self, bandwidths=None, n_jobs=1, cv=None):
        '''Doc String'''

        self._flowParams()
        self.kde.selectBandwidth(bandwidths, n_jobs, cv)
        self._flowParams(up=True)
        return self
