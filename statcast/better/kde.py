from inspect import Parameter

import numpy as np

from sklearn.neighbors.kde import KernelDensity
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.model_selection import check_cv

from scipy import stats
from scipy.special import gamma
from scipy.spatial.distance import pdist

from .base import BetterModel
from .spark import gridCVScoresAlt


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

    _params = [Parameter('normalize', Parameter.POSITIONAL_OR_KEYWORD,
                         default=True)]

    def _se(self, X):
        '''Doc String'''

        trainX = np.array(self.tree_.data)
        n = trainX.shape[0]

        Y = np.array([self._kernelFunction(X - row)
                     for row in trainX])
        s2 = Y.var(axis=0, ddof=1)
        return np.sqrt(s2 / n).flatten()

    def fit(self, X, y=None):
        '''Doc String'''

        if self.normalize:
            X = check_array(X)
            U, s, V = np.linalg.svd(X, full_matrices=False)
            self.invH_ = np.diag(np.sqrt(X.shape[0]) / s).dot(V)
            self.detH_ = 1 / np.prod(np.sqrt(X.shape[0]) / s)
            X = X.dot(self.invH_.T)

        return super().fit(X, y)

    def predict(self, X):
        ''' Doc String'''

        return np.exp(self.score_samples(X))

    def score_samples(self, X):
        '''Doc String'''

        if self.normalize:
            X = check_array(X)
            X = X.dot(self.invH_.T)
            return super().score_samples(X) - np.log(self.detH_)

        return super().score_samples(X)

    def confidence(self, X, alpha=0.05):
        '''Doc String'''

        check_is_fitted(self, ['tree_'])
        trainX = np.array(self.tree_.data)
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
            m *= self.detH_
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

        if self.normalize:
            X = check_array(X)
            X = X.dot(self.invH_.T)
            scale = self.detH_
        else:
            scale = 1
        return kernelFunctions[self.kernel](X, self.bandwidth) / scale

    def selectBandwidth(self, bandwidths=None, n_jobs=1, cv=None):
        '''Doc String'''

        check_is_fitted(self, ['tree_'])
        trainX = np.array(self.tree_.data)

        nSplits = check_cv(cv).get_n_splits()

        if trainX.shape[0] == 1:
            self.bandwidth = 1
            self.cv_results_ = None
            return self
        elif trainX.shape[0] < nSplits:
            cv = nSplits = trainX.shape[0]

        scale = ((nSplits - 1) / nSplits) ** (-1 / (4 + trainX.shape[1]))

        if bandwidths is None:
            if trainX.shape[0] > 1000:
                subs = np.random.randint(0, trainX.shape[0], size=(1000,))
                bandMax = pdist(trainX[subs]).mean()
            else:
                bandMax = pdist(trainX).mean()
            nnDists = self.tree_.query(trainX, k=2)[0][:, 1]
            if self.kernel in ['gaussian', 'exponential']:
                bandMin = nnDists.mean()
            else:
                bandMin = nnDists.max() * 1.02
            bandwidths = np.logspace(np.log10(bandMin), np.log10(bandMax),
                                     num=5)

        parameters = {'bandwidth': bandwidths * scale}
        results = gridCVScoresAlt(self, parameters, trainX,
                                  n_jobs=n_jobs, cv=cv)
        totalScores = [scores.sum() for scores in results['scores']]

        if not np.isfinite(totalScores).any():
            self.bandwidth = bandMax
        else:
            bestInd = np.argmax(totalScores)
            bestBand = results.loc[bestInd, 'params']['bandwidth']
            self.bandwidth = bestBand / scale
        self.cv_results_ = results
        return self
