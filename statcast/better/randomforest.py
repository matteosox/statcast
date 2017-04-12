import warnings
from inspect import Parameter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_array
from sklearn.metrics import mean_squared_error

from .base import BetterModel


class BetterRandomForestRegressor(RandomForestRegressor, BetterModel):
    '''Doc String'''

    @property
    def feature_importances_(self):
        '''Doc String'''

        if not self.xLabels:
            return super().feature_importances_
        ftImps = pd.Series()
        ftImpsComplete = pd.Series(super().feature_importances_,
                                   index=self.trainX_.columns). \
            sort_values(ascending=False)

        for xLabel in self.xLabels:
            ftImps[xLabel] = \
                ftImpsComplete[ftImpsComplete.index.str.startswith(xLabel)]. \
                sum()

        ftImps.sort_values(ascending=False, inplace=True)
        return ftImps

    def score(self, X, Y, sample_weight=None):
        '''Doc String'''

        Y = check_array(Y)
        Yp = self.predict(X)
        return -np.sqrt([mean_squared_error(y, yp, sample_weight)
                         for y, yp in zip(Y.T, Yp.T)]).mean()

    def createX(self, data):
        '''Doc String'''

        X = pd.DataFrame()
        for xLabel in self.xLabels:
            if data[xLabel].dtype is pd.types.dtypes.CategoricalDtype():
                for cat in data[xLabel].cat.categories:
                    X[xLabel + '|' + str(cat)] = data[xLabel] == cat
                X[xLabel + '|' + 'null'] = data[xLabel].isnull()
            else:
                X[xLabel] = data[xLabel]

        return X

    def _set_oob_score(self, X, Y):
        '''Doc String'''

        super()._set_oob_score(X, Y)
        Yp = self.oob_prediction_
        return -np.sqrt([mean_squared_error(y, yp)
                         for y, yp in zip(Y.T, Yp.T)]).mean()


class TreeSelectingRFRegressor(BetterRandomForestRegressor, BetterModel):
    '''Doc String'''

    _params = [Parameter('treeThreshold', Parameter.POSITIONAL_OR_KEYWORD,
                         default=0.99)]

    def fit(self, X, Y, sample_weight=None):
        '''Doc String'''

        warnStr = "Some inputs do not have OOB scores. " \
                  "This probably means too few trees were used " \
                  "to compute any reliable oob estimates."
        with warnings.catch_warnings(record=True) as caughtWarnings:
            warnings.filterwarnings('ignore', message=warnStr)
            self.warm_start = False
            self.oob_score = True
            self.n_estimators = 10
            scores = [super().fit(X, Y, sample_weight).oob_score_]
            nTrees = [self.n_estimators]
            for caughtWarning in caughtWarnings.copy():
                if caughtWarning.message == warnStr:
                    oobWarns = [True]
                    caughtWarnings.remove(caughtWarning)
                    break
            else:
                oobWarns = [False]
            self.warm_start = True

            while True:
                self.n_estimators = \
                    np.round(self.n_estimators * 1.2).astype(int)
                scores.append(super().fit(X, Y, sample_weight).oob_score_)
                for caughtWarning in caughtWarnings.copy():
                    if caughtWarning.message == warnStr:
                        oobWarns.append(True)
                        caughtWarnings.remove(caughtWarning)
                        break
                else:
                    oobWarns.append(False)
                nTrees.append(self.n_estimators)
                if (scores[-2] > (self.treeThreshold * scores[-1])) and \
                   (not any(oobWarns[-2:])):
                    break

        for caughtWarning in caughtWarnings:
            warnings.showwarning(caughtWarning.message, caughtWarning.category,
                                 caughtWarning.filename, caughtWarning.lineno)

        self.treeScores_ = pd.Series(scores, index=nTrees)
        return self
