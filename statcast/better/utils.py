import warnings

import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

from .spark import cross_val_score


def otherRFE(estimator, data, step=1, cv=None, scoring=None, scoreThresh=2e-2,
             n_jobs=1):
    '''Doc String'''

    intStep = step

    def score():
        estimator.fitD(data)
        return cross_val_score(estimator, estimator.createX(data),
                               estimator.createY(data), scoring=scoring,
                               cv=cv, n_jobs=n_jobs)

    scores = [score()]
    bestScore = np.mean(scores[-1])
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
        scores.append(score())
        ftImps = estimator.feature_importances_
        results = results.append(pd.DataFrame({k: v
                                               for k, v in ftImps.iteritems()},
                                              index=(0,)),
                                 ignore_index=True)
        if np.mean(scores[-1]) <= threshold:
            estimator.xLabels = list(results.columns[~results.iloc[-2, :].
                                                     isnull()])
            estimator.fitD(data)
            break
        elif np.mean(scores[-1]) > bestScore:
            bestScore = np.mean(scores[-1])
            threshold = bestScore * (1 - np.sign(bestScore) * scoreThresh)

    results.sort_values(by=0, axis=1, inplace=True)
    results['scores'] = scores
    estimator.rfeResults_ = results.iloc[:, ::-1]
    return estimator


def findTrainSplit(estimator, data, maxTrain=1.0, scoreThresh=2e-2,
                   groups=None, scoring=None, cv=None, n_jobs=1,
                   verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
    '''Doc String'''

    if isinstance(maxTrain, float) & (maxTrain == 1):
        subData = data
    else:
        subData, dummy = train_test_split(data, train_size=maxTrain)

    def score(data):
        return cross_val_score(estimator, estimator.createX(data),
                               estimator.createY(data),
                               groups, scoring, cv, n_jobs, verbose,
                               fit_params, pre_dispatch)

    scores = [score(subData)]

    def getScore(ind):
        return np.mean(scores[ind])

    def bestScore():
        return max([np.mean(score) for score in scores])

    def threshold():
        return bestScore() * (1 - np.sign(bestScore()) * scoreThresh)

    trainSizes = [subData.shape[0]]
    while trainSizes[-1] > 1:
        trainSizes.append(int(np.round(trainSizes[-1] / 2)))
        subData, dummy = train_test_split(data, train_size=trainSizes[-1])
        scores.append(score(subData))
        if getScore(-1) <= threshold():
            lowerBound = (trainSizes[-1], getScore(-1))
            upperBound = (trainSizes[-2], getScore(-2))
            break
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
        if getScore(-1) >= threshold():
            upperBound = (trainSizes[-1], getScore(-1))
            break
        else:
            lowerBound = (trainSizes[-1], getScore(-1))

    size = int(np.round(np.interp(threshold(),
                                  (lowerBound[1], upperBound[1]),
                                  (lowerBound[0], upperBound[0]))))
    subData, dummy = train_test_split(data, train_size=size)
    estimator.fitD(subData)
    estimator.trainSplitResults_ = pd.DataFrame({'size': trainSizes,
                                                 'score': scores}). \
        sort_values(by='size')
    return estimator
