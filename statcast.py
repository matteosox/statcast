import bbsavant
import pandas as pd
import numpy as np
from betterModels import treeSelectingRFRegressor, betterLME4
from betterModels import findTrainSplit, otherRFE
import gdWeather
from tools import fixPath
from plotTools import plotImages

savantDB = bbsavant.db('fast')
weatherDB = gdWeather.db('fast')
weatherData = pd.read_sql_query(
    '''SELECT *
    FROM {}'''.format(weatherDB._tblName), weatherDB.engine)


class bip():
    '''Doc String'''

    def __init__(self, years, scImputerName=None, scFactorMdlName=None):
        '''Doc String'''

        self._initData(years)
        self.scImputer = \
            treeSelectingRFRegressor(name='scImputer{}'.
                                     format('_'.join(str(year)
                                                     for year in years)),
                                     xLabels=['start_speed',
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
                                     n_estimators=100,
                                     oob_score=True)
        self.scFactorMdl = \
            betterLME4(name='scFactorMdl{}'.
                       format('_'.join(str(year)
                                       for year in years)),
                       xLabels=['batter', 'pitcher', 'gdTemp', 'home_team'],
                       yLabels=['hit_speed', 'hit_angle', 'hit_distance_sc'],
                       formulas='(1|batter) + (1|pitcher) + gdTemp + '
                       '(1|home_team)')
        if scImputerName == 'new':
            self._createSCImputer()
        elif scImputerName:
            self.scImputer = self.scImputer.load(scImputerName)
        else:
            try:
                self.scImputer = self.scImputer.load()
            except FileNotFoundError:
                self._createSCImputer()
                self.scImputer.save()
        self._imputeSCData()
        if scFactorMdlName == 'new':
            self.scFactorMdl.fitD(self.data)
        elif scFactorMdlName:
            self.scFactorMdl = self.scFactorMdl.load(scFactorMdlName)
        else:
            try:
                self.scFactorMdl = self.scFactorMdl.load()
            except FileNotFoundError:
                self.scFactorMdl.fitD(self.data)
                self.scFactorMdl.save()

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

        imputed = self.imputed
        imputeData = self.data[~self.data.exclude & imputed]
        imputeY = pd.DataFrame(self.scImputer.predictD(imputeData),
                               columns=self.scImputer.yLabels)

        for label in self.scImputer.yLabels:
            imputeThisCol = self.data.missing.map(lambda x: label in x)
            self.data.loc[~self.data.exclude & imputeThisCol, label] = \
                imputeY.loc[imputeThisCol[~self.data.exclude & imputed].values,
                            label].values

        return

    def _createSCImputer(self):
        '''Doc String'''

        imputed = self.imputed
        trainData = self.data[~self.data.exclude & ~imputed]
        self.scImputer = findTrainSplit(self.scImputer, trainData, n_jobs=-1)
        self.scImputer.n_jobs = -1
        subTrainData = trainData.loc[self.scImputer.trainX_.index, :]
        self.scImputer = otherRFE(self.scImputer, subTrainData)
        self.scImputer.n_jobs = 1
        self.scImputer = findTrainSplit(self.scImputer, trainData, n_jobs=-1)
        self.scImputer.n_jobs = -1

    @property
    def imputed(self):
        '''Doc String'''

        return self.data.missing.map(lambda x:
                                     any(y in x
                                         for y in self.scImputer.yLabels))


def plotMLBLogos(X, Y, sizes=20, alphas=1, ax=None):
    '''Doc String'''

    images = [fixPath.findFile('{}.png'.format(team.strip()))
              for team in X.index]
    thing = plotImages(X, Y, images, sizes, alphas, ax)
    if ax is None:
        ax = thing.gca()
    ax.set_xlabel(X.name)
    ax.set_ylabel(Y.name)
    return thing
