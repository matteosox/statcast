# %% Imports

from matplotlib import pyplot as plt

from statcast.bip import Bip
from statcast.tools.plot import correlationPlot
from statcast.better.utils import findTrainSplit

# %% Plot correlation of imputing model

years = (2016, 2015)

labels = ['Exit Velocity', 'Launch Angle', 'Hit Distance']
units = ['mph', 'degrees', 'feet']

for year in years:

    bip = Bip(years=(year,), n_jobs=-1)

    testData = bip.data.loc[~bip.data.exclude & ~bip.data.scImputed, :]

    testY = bip.scImputer.createY(testData)
    testYp = bip.scImputer.predictD(testData)

    labelsYr = ['{} {}'.format(label, year) for label in labels]

    figs = correlationPlot(testY,
                           testYp,
                           labels=labelsYr,
                           units=units,
                           ms=0.7)

    for fig, label in zip(figs, labels):
        fig.savefig('{} Correlation {}'.format(label, year))

# %% Plot Tree Curve

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(bip.scImputer.treeScores_.index.values,
        bip.scImputer.treeScores_.values, 'o-')
ax.set_xlabel('Number of Trees')
ax.set_ylabel('Out of Bag Score (R^2)')
ax.xaxis.set_ticklabels(ax.xaxis.get_majorticklocs().astype(int))

fig.savefig('Number of Trees Example')

# %% Plot RFE Scores

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(bip.scImputer.rfeResults_.columns.shape[0] - 1 -
        bip.scImputer.rfeResults_['scores'].index,
        [-thing.mean() for thing in bip.scImputer.rfeResults_['scores']])
ax.set_xlabel('Number of Features')
ax.set_ylabel('Cross-validation Score (RMS Error)')

fig.savefig('RFE Example')

# %% Plot Learning Curve

trainData = bip.data.loc[~bip.data.exclude & ~bip.data.scImputed, :]

findTrainSplit(bip.scImputer, trainData, cv=10, n_jobs=-1, scoreThresh=0.2)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(bip.scImputer.trainSplitResults_['size'].values,
        [-thing.mean() for thing in bip.scImputer.trainSplitResults_.score],
        'o-')
ax.set_xlabel('Number of Datapoints')
ax.set_ylabel('Cross-validation Score (RMS Error)')
ax.set_xscale('log')

fig.savefig('Learning Curve Example')
