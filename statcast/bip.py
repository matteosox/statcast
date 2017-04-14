import os

import pandas as pd
import numpy as np

from .database.bbsavant import DB as SavantDB
from .database.gd_weather import DB as WeatherDB

from .better.randomforest import TreeSelectingRFRegressor
from .better.mixed import BetterLME4
from .better.utils import findTrainSplit, otherRFE
from .tools.plot import plotKDHist
from .better.kde import BetterKernelDensity
from .better.spark import GridSearchCV

from . import __path__


savantDB = SavantDB('fast')
weatherDB = WeatherDB('fast')
weatherData = pd.read_sql_query(
    '''SELECT *
    FROM {}'''.format(weatherDB._tblName), weatherDB.engine)

_storagePath = os.path.join(__path__[0], 'Storage')

_scImputer = \
    TreeSelectingRFRegressor(xLabels=['start_speed',
                                      'x0',
                                      'z0',
                                      'events',
                                      'zone',
                                      'hit_location',
                                      'bb_type',
                                      'balls',
                                      'strikes',
                                      'pfx_x',
                                      'pfx_z',
                                      'px',
                                      'pz',
                                      'hc_x',
                                      'hc_y',
                                      'vx0',
                                      'vy0',
                                      'vz0',
                                      'effective_speed',
                                      'sprayAngle',
                                      'hitDistanceGD'],
                             yLabels=['hit_speed',
                                      'hit_angle',
                                      'hit_distance_sc'],
                             oob_score=True,
                             n_jobs=-1)
_scFactorMdl = \
    BetterLME4(xLabels=['batter', 'pitcher', 'gdTemp', 'home_team'],
               yLabels=['hit_speed', 'hit_angle', 'hit_distance_sc'],
               formulas='(1|batter) + (1|pitcher) + gdTemp + '
               '(1|home_team)')


class Bip():
    '''Doc String'''

    def __init__(self, years, scImputerName=None, scFactorMdlName=None,
                 n_jobs=-1):
        '''Doc String'''

        self.n_jobs = n_jobs
        self.years = years

        self._initData(years)

        self._initSCImputer(scImputerName=scImputerName)
        self._imputeSCData()

        self._initSCFactorMdl(scFactorMdlName=scFactorMdlName)

    def _initData(self, years):
        '''Doc String'''

        self.data = pd.DataFrame()
        for year in years:
            rawD = pd.read_sql_query(
                '''SELECT *
                FROM {}
                WHERE type = 'X'
                AND game_year = {}
                AND game_type = 'R ' '''.format(savantDB._tblName, year),
                savantDB.engine)
            self.data = self.data.append(rawD, ignore_index=True)

        self.data['sprayAngle'] = \
            (np.arctan2(208 - self.data.hc_y, self.data.hc_x - 128) /
             (2 * np.pi) * 360 + 90) % 360 - 180
        self.data['hitDistanceGD'] = np.sqrt((self.data.hc_x - 128) ** 2 +
                                             (208 - self.data.hc_y) ** 2)

        self.data[['on_3b', 'on_2b', 'on_1b']] = \
            self.data[['on_3b', 'on_2b', 'on_1b']]. \
            fillna(value=0).astype('int')
        self.data['baseState'] = \
            (self.data[['on_3b', 'on_2b', 'on_1b']] == 0). \
            replace([True, False], ['_', 'X']).sum(axis=1)

        temps = pd.Series(weatherData.temp.values, index=weatherData.game_pk)
        temps = temps[~temps.index.duplicated(keep='first')]
        self.data['gdTemp'] = temps.loc[self.data.game_pk].values

        excludeEvents = ['Batter Interference', 'Hit By Pitch', 'Strikeout',
                         'Walk', 'Fan Intereference', 'Field Error',
                         'Catcher Interference', 'Fan interference']
        self.data['exclude'] = self.data.events.isin(excludeEvents)

        categories = ['pitch_type', 'batter', 'pitcher', 'events', 'zone',
                      'stand', 'p_throws', 'home_team', 'away_team',
                      'hit_location', 'bb_type', 'on_3b', 'on_2b', 'on_1b',
                      'inning_topbot', 'catcher', 'umpire', 'game_pk',
                      'baseState']
        for category in categories:
            self.data[category] = self.data[category].astype('category')

        zeroIsMissingCols = ['hit_speed', 'hit_angle', 'hit_distance_sc']
        for col in zeroIsMissingCols:
            self.data.loc[self.data[col] == 0, col] = np.nan

        self.data['missing'] = [', '.join(self.data.columns[row])
                                for row in self.data.isnull().values]

        self.data.fillna(self.data.median(), inplace=True)
        return

    def _imputeSCData(self):
        '''Doc String'''

        imputed = self.missing(_scImputer.yLabels)
        imputeData = self.data[~self.data.exclude & imputed]
        imputeY = pd.DataFrame(self.scImputer.predictD(imputeData),
                               columns=self.scImputer.yLabels)

        for label in self.scImputer.yLabels:
            imputeThisCol = self.data.missing.map(lambda x: label in x)
            self.data.loc[~self.data.exclude & imputeThisCol, label] = \
                imputeY.loc[imputeThisCol[~self.data.exclude & imputed].values,
                            label].values

        return

    def _initSCImputer(self, scImputerName=None):
        '''Doc String'''

        if scImputerName == 'new':
            self._createSCImputer()
        elif scImputerName:
            self.scImputer = _scImputer.load(scImputerName)
        else:
            name = 'scImputer{}'.format('_'.join(str(year)
                                                 for year in self.years))
            try:
                self.scImputer = \
                    _scImputer.load(name=name, searchDirs=(_storagePath,))
            except FileNotFoundError:
                self._createSCImputer()
                self.scImputer.name = name
                self.scImputer.save(os.path.join(_storagePath,
                                                 self.scImputer.name))

    def _createSCImputer(self):
        '''Doc String'''

        imputed = self.missing(_scImputer.yLabels)
        trainData = self.data[~self.data.exclude & ~imputed]
        self.scImputer = findTrainSplit(_scImputer, trainData,
                                        n_jobs=self.n_jobs)
        subTrainData = trainData.loc[self.scImputer.trainX_.index, :]
        self.scImputer = otherRFE(self.scImputer, subTrainData, cv=10,
                                  n_jobs=self.n_jobs)
        self.scImputer = findTrainSplit(self.scImputer, trainData, cv=10,
                                        n_jobs=self.n_jobs)

    def _initSCFactorMdl(self, scFactorMdlName=None):
        '''Doc String'''

        if scFactorMdlName == 'new':
            self._createSCFactorMdl()
        elif scFactorMdlName:
            self.scFactorMdl = _scFactorMdl.load(scFactorMdlName)
        else:
            try:
                name = 'scFactorMdl{}'.format('_'.join(str(year)
                                                       for year in self.years))
                self.scFactorMdl = \
                    _scFactorMdl.load(name=name, searchDirs=(_storagePath,))
            except FileNotFoundError:
                self._createSCFactorMdl()
                self.scFactorMdl.name = name
                self.scFactorMdl.save(os.path.join(_storagePath,
                                                   self.scFactorMdl.name))

    def _createSCFactorMdl(self):
        '''Doc String'''

        imputed = self.missing(_scImputer.yLabels)
        trainData = self.data[~self.data.exclude & ~imputed]

        param_grid = {'formulas': _scFactorMdl.formulas}

        self.scFactorMdl = GridSearchCV(_scFactorMdl, param_grid). \
            fit(_scFactorMdl.createX(trainData), _scFactorMdl.createY(trainData)). \
            best_estimator_

    def missing(self, columns):
        '''Doc String'''

        return self.data.missing.map(lambda x:
                                     any(y in x
                                         for y in columns))

    def plotSCHistograms(self):
        '''Doc String'''

        labels = ['Exit Velocity', 'Launch Angle', 'Hit Distance']
        units = ['mph', 'degrees', 'feet']

        imputed = self.missing(self.scImputer.yLabels)
        inds = self.data.loc[~self.data.exclude & ~imputed, :].index
        trainInds = self.scImputer.trainX_.index
        testInds = inds.difference(trainInds)

        testData = self.data.loc[testInds, :]
        imputeData = self.data.loc[~self.data.exclude & imputed, :]

        testY = self.scImputer.createY(testData).values.T
        testYp = self.scImputer.predictD(testData).T
        imputeY = self.scImputer.predictD(imputeData).T

        name = 'bandwidths{}.csv'.format('_'.join(str(year)
                                                  for year in self.years))

        try:
            bandwidths = pd.read_csv(os.path.join(_storagePath, name),
                                     index_col=0)
        except FileNotFoundError:
            bandwidths = pd.DataFrame({'test': True,
                                       'testP': True,
                                       'impute': True},
                                      index=self.scImputer.yLabels)
            for data, col in zip([testY, testYp, imputeY],
                                 ['test', 'testP', 'impute']):
                for subData, row in zip(data, self.scImputer.yLabels):
                    xmin, xmax = min(subData), max(subData)
                    kde = BetterKernelDensity(kernel='gaussian', rtol=1e-4)
                    param_grid = {'bandwidth': np.logspace(-3, -1, num=20) *
                                  (xmax - xmin)}
                    trainGrid = \
                        GridSearchCV(kde, param_grid, cv=10,
                                     n_jobs=self.n_jobs).fit(subData[:, None])
                    bandwidths.loc[row, col] = \
                        trainGrid.best_params_['bandwidth']
                    bandwidths.to_csv(os.path.join(_storagePath, name))

        for trainy, trainyp, imputey, label, unit, yLabel in \
            zip(testY, testYp, imputeY, labels, units,
                self.scImputer.yLabels):

            fig, kde = plotKDHist(trainy, kernel='gaussian',
                                  bandwidth=bandwidths.loc[yLabel, 'test'])
            ax = fig.gca()
            plotKDHist(trainyp, kernel='gaussian', ax=ax,
                       bandwidth=bandwidths.loc[yLabel, 'testP'])
            plotKDHist(imputey, kernel='gaussian', ax=ax,
                       bandwidth=bandwidths.loc[yLabel, 'impute'])

            ax.set_xlim(left=min(trainy.min(), trainyp.min(), imputey.min()),
                        right=max(trainy.max(), trainyp.max(), imputey.max()))
            ax.set_ylim(bottom=0, auto=True)

            ax.set_xlabel(label + ' ({})'.format(unit))
            ax.legend(labels=('Test Data',
                              'Test Data Imputed',
                              'Missing Data Imputed'), loc='best')

            fig.savefig('{} {} Histogram'.
                        format(', '.join(str(year) for year in self.years),
                               label))
