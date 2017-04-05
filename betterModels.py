# %%

import abc
import numpy as np
import pandas as pd
import warnings

from scipy import stats
from inspect import signature, Signature, Parameter

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors.kde import KernelDensity
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from tools import fixPath

pandas2ri.activate()
rLME4 = importr('lme4')

addParams = [Parameter('name', Parameter.POSITIONAL_OR_KEYWORD, default=''),
             Parameter('xLabels', Parameter.POSITIONAL_OR_KEYWORD, default=()),
             Parameter('yLabels', Parameter.POSITIONAL_OR_KEYWORD, default=())]
xMethods = ['predict', 'predict_proba', 'transform', 'decision_function',
            'predict_log_proba', 'score_samples']
xyMethods = ['fit', 'fit_predict', 'fit_transform', 'score', 'partial_fit']
fitMethods = ['fit', 'fit_predict', 'fit_transform', 'partial_fit']


class betterMetaClass(abc.ABCMeta):
    ''' Doc String'''

    @staticmethod
    def addXMethod(clsobj, method):
        '''Doc String'''

        if hasattr(clsobj, method) and not hasattr(clsobj, method + 'D'):
            methodSig = signature(getattr(clsobj, method))
            methodParams = [p for p in methodSig.parameters.values()]
            methodDParams = methodParams.copy()
            if methodDParams[0].name != 'self':
                methodDParams.insert(0,
                                     Parameter('self',
                                               Parameter.POSITIONAL_ONLY))
            if methodDParams[1].kind is Parameter.VAR_POSITIONAL:
                methodDParams.insert(1, Parameter('data',
                                                  Parameter.
                                                  POSITIONAL_OR_KEYWORD))
            else:
                methodDParams[1] = \
                    Parameter('data', Parameter.POSITIONAL_OR_KEYWORD)

            methodDSig = Signature(methodDParams)

            def methodD(self, data, *args, **kwargs):
                X = self.createX(data)
                return getattr(self, method)(X, *args, **kwargs)
            methodD.__signature__ = methodDSig
            setattr(clsobj, method + 'D', methodD)

    @staticmethod
    def addXYMethod(clsobj, method):
        '''Doc String'''

        if hasattr(clsobj, method) and not hasattr(clsobj, method + 'D'):
            methodSig = signature(getattr(clsobj, method))
            methodParams = [p for p in methodSig.parameters.values()]
            methodDParams = methodParams.copy()
            if methodDParams[0].name != 'self':
                methodDParams.insert(0,
                                     Parameter('self',
                                               Parameter.POSITIONAL_ONLY))
            if methodDParams[1].kind is Parameter.VAR_POSITIONAL:
                methodDParams.insert(1, Parameter('data',
                                                  Parameter.
                                                  POSITIONAL_OR_KEYWORD))
            else:
                methodDParams[1] = \
                    Parameter('data', Parameter.POSITIONAL_OR_KEYWORD)
            if methodDParams[2] is Parameter.VAR_POSITIONAL:
                pass
            else:
                del methodDParams[2]
            methodDSig = Signature(methodDParams)

            def methodD(self, data, *args, **kwargs):
                X = self.createX(data)
                Y = self.createY(data)
                return getattr(self, method)(X, Y, *args, **kwargs)
            methodD.__signature__ = methodDSig
            setattr(clsobj, method + 'D', methodD)

    @staticmethod
    def storeTrainData(clsobj, methodName):
        '''Doc String'''

        if hasattr(clsobj, methodName):
            method = getattr(clsobj, methodName)
            methodSig = signature(method)

            def newMethod(self, X, Y=None, *args, **kwargs):
                '''Doc String'''

                self = method(self, X, Y, *args, **kwargs)
                self.trainX_ = X.copy()
                if Y is not None:
                    self.trainY_ = Y.copy()
                return self
            newMethod.__signature__ = methodSig
            setattr(clsobj, methodName, newMethod)

    def __new__(cls, clsname, bases, clsdict):
        '''Doc String'''

        clsobj = super().__new__(cls, clsname, bases, clsdict)
        oldInit = clsobj.__init__
        oldParams = [param for param in signature(oldInit).parameters.values()]

        firstIn = oldParams.pop(0)
        if firstIn.name != 'self':
            raise RuntimeError('betterModels init must have self first '
                               'argument')
        if oldInit is object.__init__:
            oldParams = []
        if addParams == oldParams[:len(addParams)]:
            newParams = []
        else:
            newParams = addParams
        if any(nP == oP for nP in newParams for oP in oldParams):
            raise RuntimeError('betterModels cannot define {} as inputs to '
                               'init'.format(', '.join([nP.name
                                                        for nP in newParams])))
        try:
            if any(p.kind != Parameter.POSITIONAL_OR_KEYWORD
                   for p in clsobj._params):
                raise RuntimeError('betterModels can only add positional or '
                                   'keyword inputs to init from _params')
            elif any(p.default is Parameter.empty
                     for p in clsobj._params):
                raise RuntimeError('betterModels init inputs must have '
                                   'defaults')
        except AttributeError:
            raise RuntimeError('betterModels _params class attribute must be '
                               'a list of Parameters from the inspect module')
        newSig = Signature([firstIn] + newParams + clsobj._params + oldParams)

        def newInit(self, *args, **kwargs):
            '''Doc String'''

            bound = newSig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            bound.arguments.popitem(False)
            for dummy in range(len(newParams)):
                name, val = bound.arguments.popitem(False)
                setattr(self, name, val)
            for dummy in range(len(clsobj._params)):
                name, val = bound.arguments.popitem(False)
                setattr(self, name, val)

            oldInit(self, *bound.args, **bound.kwargs)

        newInit.__signature__ = newSig
        setattr(clsobj, '__init__', newInit)

        for method in fitMethods:
            cls.storeTrainData(clsobj, method)

        for method in xMethods:
            cls.addXMethod(clsobj, method)

        for method in xyMethods:
            cls.addXYMethod(clsobj, method)

        return clsobj


class betterModel(metaclass=betterMetaClass):
    '''Doc String'''

    _params = []

    def createX(self, data):
        '''Doc String'''

        return data[self.xLabels]

    def createY(self, data):
        '''Doc String'''

        return data[self.yLabels]

    def save(self, path=None):
        '''Doc String'''

        if path is None:
            path = self.name
        joblib.dump(self, path + '.pkl')

    def load(self, path=None):
        '''Doc String'''

        if path is None:
            path = self.name + '.pkl'
        path = fixPath.findFile(path)
        if not path:
            raise FileNotFoundError('Could not find {} on path'.format(path))
        return joblib.load(path)

# %%


class betterRandomForestRegressor(RandomForestRegressor, betterModel):
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


class treeSelectingRFRegressor(betterRandomForestRegressor, betterModel):
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

# %%


def otherRFE(estimator, data, step=1, scoreThresh=2e-2):
    '''Doc String'''

    intStep = step
    bestScore = estimator.fitD(data).oob_score_
    scores = [bestScore]
    threshold = bestScore * (1 - np.sign(bestScore) * scoreThresh)
    ftImps = estimator.feature_importances_
    results = pd.DataFrame({k: v for k, v in ftImps.iteritems()}, index=(0,))
    while True:
        if len(ftImps.index) == 1:
            break
        if isinstance(step, float):
            intStep = int(min(np.floor(len(ftImps.index) * step), 1))
        if intStep > len(ftImps.index):
            estimator.xLabels = list(ftImps.index[:-1])
        else:
            estimator.xLabels = list(ftImps.index[:-intStep])
        scores.append(estimator.fitD(data).oob_score_)
        ftImps = estimator.feature_importances_
        results = results.append(pd.DataFrame({k: v
                                               for k, v in ftImps.iteritems()},
                                              index=(0,)),
                                 ignore_index=True)
        if scores[-1] <= threshold:
            estimator.xLabels = list(results.columns[~results.iloc[-2, :].
                                                     isnull()])
            estimator.fitD(data)
            break
        elif scores[-1] > bestScore:
            bestScore = scores[-1]
            threshold = bestScore * (1 - np.sign(bestScore) * scoreThresh)

    results.sort_values(by=0, axis=1, inplace=True)
    results['scores'] = scores
    estimator.rfeResults_ = results.iloc[:, ::-1]
    return estimator

# %%


def findTrainSplit(estimator, data, maxTrain=1.0, scoreThresh=2e-2,
                   groups=None, scoring=None, cv=None, n_jobs=1,
                   verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
    '''Doc String'''

    if isinstance(maxTrain, float) & (maxTrain == 1):
        subData = data
    else:
        subData, dummy = train_test_split(data, train_size=maxTrain)

    def score(data):
        return np.mean(cross_val_score(estimator, estimator.createX(data),
                                       estimator.createY(data),
                                       groups, scoring, cv, n_jobs, verbose,
                                       fit_params, pre_dispatch))
    bestScore = score(subData)
    threshold = bestScore * (1 - np.sign(bestScore) * scoreThresh)
    scores = [bestScore]
    trainSizes = [subData.shape[0]]
    while trainSizes[-1] > 1:
        trainSizes.append(int(np.round(trainSizes[-1] / 2)))
        subData, dummy = train_test_split(data, train_size=trainSizes[-1])
        scores.append(score(subData))
        if scores[-1] <= threshold:
            lowerBound = (trainSizes[-1], scores[-1])
            upperBound = (trainSizes[-2], scores[-2])
            break
        elif scores[-1] > bestScore:
            bestScore = scores[-1]
            threshold = bestScore * (1 - np.sign(bestScore) * scoreThresh)
    else:
        warnings.warn('Training size of 1 found')
        estimator.fitD(subData)
        estimator.trainSplitResults_ = pd.DataFrame({'size': trainSizes,
                                                     'score': scores}). \
            sort_values(by='size')
        return estimator

    for dummy in range(3):
        trainSizes.append(int(np.round(trainSizes[-1] * 2 ** (1/4))))
        subData, dummy = train_test_split(data, train_size=trainSizes[-1])
        scores.append(score(subData))
        if scores[-1] >= threshold:
            upperBound = (trainSizes[-1], scores[-1])
            break
        else:
            lowerBound = (trainSizes[-1], scores[-1])

    size = int(np.round(np.interp(threshold,
                                  (lowerBound[1], upperBound[1]),
                                  (lowerBound[0], upperBound[0]))))
    subData, dummy = train_test_split(data, train_size=size)
    estimator.fitD(subData)
    estimator.trainSplitResults_ = pd.DataFrame({'size': trainSizes,
                                                 'score': scores}). \
        sort_values(by='size')
    return estimator

# %%


class betterLME4(BaseEstimator, RegressorMixin, betterModel):

    _params = [Parameter('formulas', Parameter.POSITIONAL_OR_KEYWORD,
                         default=()),
               Parameter('LME4Params', Parameter.POSITIONAL_OR_KEYWORD,
                         default={})]

    def fit(self, X, Y):
        '''Doc String'''

        check_X_y(X, Y, multi_output=True, dtype=None)
        data = pd.concat((pd.DataFrame(X, columns=self.xLabels),
                          pd.DataFrame(Y, columns=self.yLabels)), axis=1)
        return self.fitD(data)

    def fitD(self, data):
        '''Doc String'''

        if not any((self.xLabels, self.yLabels, self.formulas)):
            raise RuntimeError('betterLME must have xLabels, yLabels, and '
                               'formulas defined')
        if isinstance(self.formulas, str):
            self.formulas = (self.formulas,)
        if isinstance(self.yLabels, str):
            self.yLabels = (self.yLabels) * len(self.formulas)
        if len(self.formulas) != len(self.yLabels):
            if len(self.formulas) == 1:
                self.formulas = self.formulas * len(self.yLabels)
            else:
                raise RuntimeError('formulas must be a single string, or a '
                                   'tuple the same length as yLabels')
        X, Y = self.createX(data), self.createY(data)
        check_X_y(X, Y, multi_output=True, dtype=None)
        self.models_ = {}
        self.factors_ = {}
        for ii, yLabel in enumerate(self.yLabels):
            subData = pd.concat((X, Y[yLabel]), axis=1)
            formula = yLabel + ' ~ ' + self.formulas[ii]
            model = rLME4.lmer(formula=formula,
                               data=subData,
                               **self.LME4Params)
            self.models_[yLabel] = model
            self.factors_[yLabel] = self._factor(model)

        return self

    def createX(self, data):
        '''Doc String'''

        X = data[self.xLabels].copy()
        for xLabel in self.xLabels:
            if X[xLabel].dtype is pd.types.dtypes.CategoricalDtype():
                X[xLabel] = X[xLabel].astype(X[xLabel].cat.categories.dtype)

        return X

    @staticmethod
    def _factor(model):
        '''Doc String'''

        rEffs = rLME4.random_effects(model)
        fEffs = rLME4.fixed_effects(model)
        factor = {}

        for elem, name in zip(rEffs, rEffs.names):
            factor[name] = pandas2ri.ri2py(elem)

        for elem, name in zip(fEffs, fEffs.names):
            factor[name] = elem

        return factor

    def predictD(self, data):
        '''Doc String'''

        check_is_fitted(self, ['models_', 'factors_'])

        X = self.createX(data)
        check_array(X, dtype=None)

        Y = pd.DataFrame()
        for yLabel in self.yLabels:
            Y[yLabel] = \
                pandas2ri.ri2py(rLME4.predict_merMod(self.models_[yLabel],
                                                     newdata=X,
                                                     allow_new_levels=True))

        return Y

    def predict(self, X):
        '''Doc String'''

        check_array(X, dtype=None)
        return self.predictD(pd.DataFrame(X, columns=self.xLabels))

    def score(self, X, Y, sample_weight=None):
        '''Doc String'''

        Y = check_array(Y)
        Yp = self.predict(X)
        return -np.sqrt([mean_squared_error(y, yp, sample_weight)
                         for y, yp in zip(Y.T, Yp.values.T)]).mean()

    def chooseFormula(self, data, formulas, fullOut=False):
        '''Doc String'''

        pastREML = self.LME4Params.get('REML', None)
        self.LME4Params['REML'] = False

        scores = pd.DataFrame()
        for formula in formulas:
            self.formulas = formula
            self.fitD(data)
            score = {}
            for name, mdl in self.models_.items():
                score[name] = pandas2ri.ri2py(rLME4.llikAIC(mdl)[1])[1]
            scores = scores.append(pd.DataFrame(score, index=(0,)),
                                   ignore_index=True)

        if pastREML is None:
            del self.LME4Params['REML']
        else:
            self.LME4Params['REML'] = pastREML

        self.formulas = \
            tuple(formulas[scores[yLabel].idxmin()] for yLabel in self.yLabels)

        self.fitD(data)
        self.scores_ = scores
        return self

# %%


class betterKernelDensity(KernelDensity, betterModel):
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
