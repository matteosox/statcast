import abc

from inspect import Parameter

import numpy as np

from statsmodels import api as sm

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .base import BetterModel


class BetterSM(BaseEstimator, metaclass=abc.ABCMeta):

    _params = [Parameter('addConstant', Parameter.POSITIONAL_OR_KEYWORD,
                         default=True),
               Parameter('SMParams', Parameter.POSITIONAL_OR_KEYWORD,
                         default={})]

    @abc.abstractmethod
    def mdlClass():
        pass

    def fit(self, X, y):
        '''Doc String'''

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.addConstant:
            X = sm.tools.tools.add_constant(X)
        self.mdl_ = self.mdlClass(y, X, **self.SMParams)
        self.results_ = self.mdl_.fit()
        return self

    def predict_proba(self, X):
        '''Doc String'''

        check_is_fitted(self, ['mdl_', 'results_'])
        check_array(X, dtype=None)

        if self.addConstant:
            X = sm.tools.tools.add_constant(X)

        return self.results_.predict(X)

    def predict(self, X):
        '''Doc String'''

        probs = self.predict_proba(X)
        y = self.classes_.take(probs.argmax(1))
        return y


class BetterGLM(BetterSM, BetterModel):

    mdlClass = sm.GLM


class BetterMNLogit(BetterSM, BetterModel):

    mdlClass = sm.MNLogit
