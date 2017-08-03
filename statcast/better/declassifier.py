import abc
import warnings
from inspect import Parameter

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.fixes import bincount
from sklearn.utils.validation import check_is_fitted

from .base import BetterModel
from .kde import BetterKernelDensity


class DensityEstimationClassifier(BaseEstimator, ClassifierMixin,
                                  metaclass=abc.ABCMeta):

    _params = [Parameter('priors', Parameter.POSITIONAL_OR_KEYWORD,
                         default=None)]

    @abc.abstractmethod
    def fit(self, X, y):
        '''Doc String'''

        pass

    def _prefit(self, X, y):
        '''Doc String'''

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if self.priors_.sum() != 1:
            warnings.warn("The priors do not sum to 1. Renormalizing",
                          UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        return X, y

    @abc.abstractmethod
    def _estimateDensities(self, X):
        '''Doc String'''

        pass

    def predict_proba(self, X):
        '''Doc String'''

        X = check_array(X)
        check_is_fitted(self, ['priors_', 'classes_'])
        F = self._estimateDensities(X)
        num = F * self.priors_
        den = num.sum(1, keepdims=True)
        den0 = den == 0
        den[den0] = 1
        probs = num / den
        probs[den0.flatten()] = self.priors_
        return probs

    def predict(self, X):
        '''Doc String'''

        probs = self.predict_proba(X)
        y = self.classes_.take(probs.argmax(1))
        return y


class KDEClassifier(DensityEstimationClassifier, BetterModel):

    _params = DensityEstimationClassifier._params
    _params.extend([Parameter('kdeParams', Parameter.POSITIONAL_OR_KEYWORD,
                              default={}),
                    Parameter('cv', Parameter.POSITIONAL_OR_KEYWORD,
                              default=None),
                    Parameter('n_jobs', Parameter.POSITIONAL_OR_KEYWORD,
                              default=1)])

    def fit(self, X, y):
        '''Doc String'''

        X, y = self._prefit(X, y)
        nClasses = len(self.classes_)
        self.kdes_ = []
        for i in range(nClasses):
            kde = BetterKernelDensity(**self.kdeParams)
            kde.fit(X[y == i, :])
            kde.selectBandwidth(n_jobs=self.n_jobs, cv=self.cv)
            self.kdes_.append(kde)

        return self

    def _estimateDensities(self, X):
        '''Doc String'''

        return np.concatenate([kde.predict(X)[:, None] for kde in self.kdes_],
                              axis=1)
