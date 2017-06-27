from inspect import signature

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors.kde import KernelDensity
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.metrics import mean_squared_error

from scipy import stats

from .base import BetterModel
from .kde import BetterKernelDensity
from .spark import GridSearchCV


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
        return self

    def _flowParams(self):
        '''Doc String'''

        self.kde.set_params(**self.get_params())

    def _weights(self, X):
        '''Doc String'''

        self._flowParams()
        num = np.hstack([self.kde._kernelFunction(np.tile(
            row, (self.trainX_.shape[0], 1)) - self.trainX_)[:, None]
            for row in X]).T
        den = np.tile(self.kde.predict(X)[:, None] *
                      self.trainX_.shape[0], (1, self.trainX_.shape[0]))
        return num / den

    def predict(self, X):
        '''Doc String'''

        check_is_fitted(self, ['trainX_', 'trainY_'])
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

        self._flowParams()
        k0 = self.kde._kernelFunction(np.zeros((1, self.trainX_.shape[1])))
        den = np.tile(1 - k0 / (self.kde.predict(self.trainX_)[:, None] *
                                self.trainX_.shape[0]),
                      (1, self.trainY_.shape[1]))
        num = self.trainY_ - self.predict(self.trainX_)
        return ((num / den) ** 2).mean(0)

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
        se = np.sqrt((self._weights(X) ** 2).sum(1, keepdims=True) *
                     self.risk())
        return f - q * se, f + q * se

    def confidenceD(self, data, alpha=0.05):
        '''Doc String'''

        X = self.createX(data)
        return self.confidence(X, alpha)

    def selectBandwidth(self, bandwidths=None, n_jobs=1, cv=None):
        '''Doc String'''

        if bandwidths is None:
            xmins, xmaxs = self.trainX_.min(0), self.trainX_.max(0)
            bandwidths = np.logspace(-3, -1, num=10) * (xmaxs - xmins).max()

        # Leave one out cross-validation
        if cv == -1:
            risks = np.array([self.set_params(bandwidth=bandwidth).risk()
                              for bandwidth in bandwidths])
            self.bandwidth = bandwidths[np.argmin(risks)]
            self.cv_results_ = pd.DataFrame({'bandwidth': bandwidths,
                                             'risk': risks})
            return self

        parameters = {'bandwidth': bandwidths}
        trainGrid = GridSearchCV(self, parameters, cv=cv,
                                 n_jobs=n_jobs, refit=False).fit(self.trainX_)
        self.bandwidth = trainGrid.best_estimator_.bandwidth
        self.cv_results_ = trainGrid.cv_results_
        return self
