#%%

%reset -f

from tools import fixPath

fixPath()

#%%

import bbsavant
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from kernelDensityConfidence import KDConfidence
from sklearn.model_selection import GridSearchCV
from tools import correlationPlot
import gdWeather
from scipy import stats
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


#%% Load raw 2016 regular season baseball savant data

db = bbsavant.db()

rawD = pd.read_sql_query(
    '''SELECT *
    FROM {}
    WHERE type is 'X'
    AND game_year IS 2015
    AND game_type IS 'R ' '''.format(db._tblName), db.engine)

 #%%

weatherDB = gdWeather.db()

weatherData = pd.read_sql_query(
    '''SELECT *
    FROM {}'''.format(weatherDB._tblName), weatherDB.engine)

#%% Create clean copy of the data

data = rawD.copy()

data['sprayAngle'] = (np.arctan2(208 - data.hc_y, data.hc_x - 128) /
                      (2 * np.pi) * 360 + 90) % 360 - 180
data['hitDistanceGD'] = np.sqrt((data.hc_x - 128) ** 2 +
                                (208 - data.hc_y) ** 2)

data[['on_3b', 'on_2b', 'on_1b']] = \
    data[['on_3b', 'on_2b', 'on_1b']].fillna(value=0).astype('int')
data['baseState'] = (data[['on_3b', 'on_2b', 'on_1b']] == 0). \
    replace([True, False], ['_', 'X']).sum(axis=1)


temps = pd.Series(weatherData.temp.values, index=weatherData.game_pk)
temps = temps[~temps.index.duplicated(keep='first')]
data['gdTemp'] = temps.loc[data.game_pk].values

excludeEvents = ['Batter Interference', 'Hit By Pitch', 'Strikeout', 'Walk',
                 'Fan Intereference', 'Field Error', 'Catcher Interference',
                 'Fan interference']
data['exclude'] = data.events.isin(excludeEvents)

categories = ['pitch_type', 'batter', 'pitcher', 'events', 'zone', 'stand',
              'p_throws', 'home_team', 'away_team', 'hit_location', 'bb_type',
              'on_3b', 'on_2b', 'on_1b', 'inning_topbot', 'catcher', 'umpire',
              'game_pk', 'baseState']
for category in categories:
    data[category] = data[category].astype('category')

zeroIsMissingCols = ['hit_speed', 'hit_angle', 'hit_distance_sc']
for col in zeroIsMissingCols:
    data.loc[data[col] == 0, col] = np.nan

data['missing'] = [', '.join(data.columns[row]) for row in data.isnull().values]

data.fillna(data.median(), inplace=True)

#%% Setup training and imputing data

xLabels = ['hitDistanceGD', 'bb_type', 'sprayAngle', 'events', 'hc_x', 'hc_y',
           'hit_location']

yLabels = ['hit_speed', 'hit_angle', 'hit_distance_sc']

X = pd.DataFrame()
for xLabel in xLabels:
    if data[xLabel].dtype is pd.types.dtypes.CategoricalDtype():
        for cat in data[xLabel].cat.categories:
            X[xLabel + '|' + str(cat)] = data[xLabel] == cat
        X[xLabel + '|' + 'null'] = data[xLabel].isnull()
    else:
        X[xLabel] = data[xLabel]

imputeSC = data.missing.map(lambda x: any(y in x for y in yLabels))
imputeX = X[~data.exclude & imputeSC]
trainX = X[~data.exclude & ~imputeSC]
trainY = data.loc[~data.exclude & ~imputeSC, yLabels]

#%% Train model and impute missing data

regressor = RandomForestRegressor(n_estimators=30, max_features='auto',
                                  max_depth=None, min_samples_split=2,
                                  oob_score=True, n_jobs=-1).fit(trainX,
                                                                 trainY)

trainYp = regressor.oob_prediction_
imputeY = regressor.predict(imputeX)

for ii, yLabel in enumerate(yLabels):
    data[yLabel + 'p'] = pd.Series(trainYp[:, ii],
                                   index=data.index[~data.exclude & ~imputeSC])
    imputeThisCol = data.missing.map(lambda x: yLabel in x)
    data.loc[~data.exclude & imputeThisCol, yLabel] = \
        imputeY[imputeThisCol[~data.exclude & imputeSC].values, ii]
    data.loc[~data.exclude & imputeSC, yLabel + 'p'] = imputeY[:, ii]

#%% Test out n_estimators

results = pd.DataFrame()

for n_estimators in np.logspace(1,np.log10(200),20).round().astype('int'):
    regressor = RandomForestRegressor(n_estimators=n_estimators,
                                      max_features='auto',
                                      max_depth=None, min_samples_split=2,
                                      oob_score=True, n_jobs=-1).fit(trainX,
                                                                     trainY)
    trainYp = regressor.oob_prediction_

    for y, yp, label in zip(trainY.values.T, trainYp.T, yLabels):
        rmsErr = np.sqrt(np.mean((y - yp) ** 2))
        r2 = stats.spearmanr(y, yp)[0] ** 2
        mae = np.mean(np.abs(y - yp))
        results = results.append(pd.DataFrame({'n_estimators': n_estimators,
                                     'label': label,
                                     'rmsErr': rmsErr,
                                     'r2': r2,
                                     'mae': mae}, index=[0]),
                       ignore_index=True)

#%% 

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

for label in yLabels:
    ax1.plot(results[results.label == label].n_estimators,
            results[results.label == label].mae, label=label)
    ax2.plot(results[results.label == label].n_estimators,
            results[results.label == label].r2, label=label)
    ax3.plot(results[results.label == label].n_estimators,
            results[results.label == label].rmsErr, label=label)

ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0, top=1)
ax3.set_ylim(bottom=0)

ax3.set_xlabel('n_estimators')
ax1.set_ylabel('Mean Absolute Error')
ax2.set_ylabel('R Squared')
ax3.set_ylabel('RMS Error')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=len(yLabels), frameon=False)
fig.suptitle('Evaluating n_estimators impact on model quality')

#%% Plot predictions

yFancyLabels = ['Exit Velocity', 'Launch Angle', 'Hit Distance']
yUnits = ['mph', 'degrees', 'feet']

figs = correlationPlot(trainY, trainYp, labels=yFancyLabels, units=yUnits)
#[fig.savefig(yLabel + ' 2015ImputerResultsOnly') for fig, yLabel in zip(figs, yFancyLabels)]

#%% Feature Importances

ftImps = pd.Series()
ftImpsComplete = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)

for xLabel in xLabels:
    ftImps[xLabel] = ftImpsComplete[ftImpsComplete.index.str.startswith(xLabel)].sum()

ftImps.sort_values(ascending=False, inplace=True)

print(ftImps)

#%% Estimate & plot histograms, NEEDS EDITING

alpha = 5e-2

for ii, (yFancy, yUnit) in enumerate(zip(yFancyLabels, yUnits)):
    trainy = trainY.values[:, ii][:, None]
    trainyp = trainYp[:, ii][:, None]
    imputey = imputeY[:, ii][:, None]

    xmin = min((trainy.min(), trainyp.min(), imputey.min()))
    xmax = max((trainy.max(), trainyp.max(), imputey.max()))
    xFit = np.linspace(xmin, xmax, 1e3)

    kde = KDConfidence(kernel='epanechnikov', rtol=1e-4)
    parameters = {'bandwidth': np.logspace(-3, -1, num=20) * (xmax - xmin)}
    trainGrid = GridSearchCV(kde, parameters, cv=10, n_jobs=-1).fit(trainy)
    trainpGrid = GridSearchCV(kde, parameters, cv=10, n_jobs=-1).fit(trainyp)
    imputeGrid = GridSearchCV(kde, parameters, cv=10, n_jobs=-1).fit(imputey)
    trainKDE = trainGrid.best_estimator_
    trainpKDE = trainpGrid.best_estimator_
    imputeKDE = imputeGrid.best_estimator_
#    trainKDE = KDConfidence(**imputeKDE.get_params()).fit(trainy)
#    trainpKDE = KDConfidence(**imputeKDE.get_params()).fit(trainyp)

    trainFitL, trainFitU = trainKDE.confidence(xFit[:, None], alpha=alpha)
    trainpFitL, trainpFitU = trainpKDE.confidence(xFit[:, None], alpha=alpha)
    imputeFitL, imputeFitU = imputeKDE.confidence(xFit[:, None], alpha=alpha)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.fill_between(xFit, trainFitU, trainFitL,
                    label='Training Data', alpha=0.35, lw=0)
    ax.fill_between(xFit, trainpFitU, trainpFitL,
                    label='Training Data Imputed', alpha=0.35, lw=0)
    ax.fill_between(xFit, imputeFitU, imputeFitL,
                    label='Missing Data Imputed', alpha=0.35, lw=0)

    ax.set_xlabel(yFancy + ' ({})'.format(yUnit))
    ax.set_ylabel('Probability Density (%)')
    ax.legend()

    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=0)

#    fig.savefig('{} Histogram'.format(yFancy))

#%% Investigate Biases

pandas2ri.activate()
rBase = importr('base')
rLME4 = importr('lme4')

formula = '{y} ~ (1|batter) + (1|pitcher) + gdTemp + (1|home_team)'

parkFactors = pd.DataFrame()
batterFactors = pd.DataFrame()
pitcherFactors = pd.DataFrame()
tempFactors = pd.Series()

for yLabel in yLabels:
    imputeThisCol = data.missing.map(lambda x: yLabel in x)
    subData = data.loc[~data.exclude & ~imputeThisCol,
                       [yLabel, 'batter', 'pitcher', 'gdTemp', 'home_team']]
    subData.pitcher = subData.pitcher.astype(int)
    subData.batter = subData.batter.astype(int)
    subData.home_team = subData.home_team.astype(str)

    mdl = rLME4.lmer(formula=formula.format(y=yLabel), data=subData)
    print(rBase.summary(mdl))
    rEffs = rLME4.random_effects(mdl)
    fEffs = rLME4.fixed_effects(mdl)

    mdlParams = {}

    for elem, name in zip(rEffs, rEffs.names):
        mdlParams[name] = pandas2ri.ri2py(elem)

    for elem, name in zip(fEffs, fEffs.names):
        mdlParams[name] = elem

    parkFactors[yLabel] = mdlParams['home_team'].iloc[:,0]
    batterFactors[yLabel] = mdlParams['batter'].iloc[:,0]
    pitcherFactors[yLabel] = mdlParams['pitcher'].iloc[:,0]
    tempFactors[yLabel] = mdlParams['gdTemp']
