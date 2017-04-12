# %% Fixpath, imports

import pandas as pd

from statcast.bip import Bip
from statcast.tools.plot import correlationPlot, plotKDHist


# %% Load data

years = (2016,)

bip = Bip(years=years)

# %% Plot Imputer Results

labels = ['Exit Velocity', 'Launch Angle', 'Hit Distance']
units = ['mph', 'degrees', 'feet']

figs = correlationPlot(bip.scImputer.trainY_,
                       bip.scImputer.oob_prediction_,
                       labels=labels,
                       units=units,
                       ms=1)

[fig.savefig(label + ' {} ImputerResultsOnly'.format(' ,'.join(years)))
    for fig, label in zip(figs, labels)]

print(bip.scImputer.feature_importances_)

# %% Compare histograms of training, imputed training, and imputed missing data

imputed = bip.imputed
trainY = bip.data.loc[~bip.data.exclude & ~imputed,
                      bip.scImputer.yLabels].values.T
trainYp = bip.scImputer.model.oob_prediction_.T
imputeY = bip.data.loc[~bip.data.exclude & imputed,
                       bip.scImputer.yLabels].values.T

for trainy, trainyp, imputey, label, unit in zip(trainY,
                                                 trainYp,
                                                 imputeY,
                                                 labels,
                                                 units):

    fig, = plotKDHist(trainy, kernel='gaussian')
    ax = fig.gca()
    plotKDHist(trainyp, kernel='gaussian', ax=ax)
    plotKDHist(imputey, kernel='gaussian', ax=ax)

    ax.set_xlabel(label + ' ({})'.format(unit))
    ax.legend(labels=('Training Data',
                      'Training Data Imputed',
                      'Missing Data Imputed'), loc='best')

    ax.set_xlim(left=min(trainy.min(), trainyp.min(), imputey.min()),
                right=max(trainy.max(), trainyp.max(), imputey.max()))
    ax.set_ylim(bottom=0)

    fig.savefig('{} Histogram'.format(label))

# %% Save Park Factors

parkFactors = \
    pd.DataFrame({label:
                  bip.scFactorMdl.factors[label]['home_team'].iloc[:, 0]
                  for label in bip.scFactorMdl.yLabels})
parkFactors.to_csv('Park Factors 2015-2016.csv')

# %% Compare 2015 & 2016 Park Factors

data2015 = bip.data[bip.data['game_year'] == 2015]
data2016 = bip.data[bip.data['game_year'] == 2016]

scFactorMdl15 = statcast.scFactorMdl(data=data2015, dump=False)
scFactorMdl16 = statcast.scFactorMdl(data=data2016, dump=False)

for key, label, unit in zip(bip.scFactorMdl.yLabels, labels, units):
    X = scFactorMdl15.factors[key]['home_team'].iloc[:, 0]
    X.name = '2015 {} ({})'.format(label, unit)
    Y = scFactorMdl16.factors[key]['home_team'].iloc[:, 0]
    Y.name = '2016 {} ({})'.format(label, unit)
    fig, = correlationPlot(X, Y, mfc='None', mec='None')
    statcast.plotMLBLogos(X, Y, ax=fig.gca())
    fig.savefig('{} Park Factor 15-16 Correlation'.format(label))

# %% Compare Observed and Imputed Park Factors

dataObs = bip.data[~imputed]
dataImp = bip.data[imputed]

scFactorMdlObs = statcast.scFactorMdl(data=dataObs, dump=False)
scFactorMdlImp = statcast.scFactorMdl(data=dataImp, dump=False)

for key, label, unit in zip(bip.scFactorMdl.yLabels, labels, units):
    X = scFactorMdlObs.factors[key]['home_team'].iloc[:, 0]
    X.name = 'Observed {} ({})'.format(label, unit)
    Y = scFactorMdlImp.factors[key]['home_team'].iloc[:, 0]
    Y.name = 'Imputed {} ({})'.format(label, unit)
    fig, = correlationPlot(X, Y, mfc='None', mec='None')
    statcast.plotMLBLogos(X, Y, ax=fig.gca())
    fig.savefig('{} Park Factor Observed-Imputed Correlation'.format(label))

# %% Compare 2015 & 2016 Player Skills
