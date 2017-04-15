# %% Fixpath, imports

import pandas as pd
from pyspark import SparkContext

from statcast.bip import Bip
from statcast.tools.plot import correlationPlot


# %% Load data

sc = SparkContext(appName="blogPost3")
years = (2016,)
bip = Bip(years=years, n_jobs=sc)

# %% Plot Imputer Results

imputed = bip.missing(bip.scImputer.yLabels)
inds = bip.data.loc[~bip.data.exclude & ~imputed, :].index
trainInds = bip.scImputer.trainX_.index
testInds = inds.difference(trainInds)

testData = bip.data.loc[testInds, :]

testY = bip.scImputer.createY(testData)
testYp = bip.scImputer.predictD(testData)

labels = ['Exit Velocity', 'Launch Angle', 'Hit Distance']
units = ['mph', 'degrees', 'feet']

figs = correlationPlot(testY,
                       testYp,
                       labels=labels,
                       units=units,
                       ms=1)

[fig.savefig(label + ' {} ImputerResultsOnly'.format(
    ' ,'.join(str(year) for year in years)))
    for fig, label in zip(figs, labels)]

print(bip.scImputer.feature_importances_)

# %% Compare histograms of testing, imputed testing, and imputed missing data

bip.plotSCHistograms()

# %% Save Park Factors

parkFactors = \
    pd.DataFrame({label:
                  bip.scFactorMdl.factors[label]['home_team'].iloc[:, 0]
                  for label in bip.scFactorMdl.yLabels})
parkFactors.to_csv('Park Factors 2015-2016.csv')

sc.stop()
