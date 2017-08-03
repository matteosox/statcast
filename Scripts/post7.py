# %% Imports

from scipy import stats
from matplotlib import pyplot as plt

from statcast.bip import Bip
from statcast.plot import plotMLBLogos
from statcast.tools.plot import addText


# %%

bip15 = Bip(years=(2015,), n_jobs=-1)
bip16 = Bip(years=(2016,), n_jobs=-1)

# %% Plot Correlations

labels = ['hit_speed', 'hit_angle', 'hit_distance_sc']
units = ['mph', 'degrees', 'feet']
fancyLabels = ['Exit Velocity', 'Launch Angle', 'Hit Distance']

for label, unit, fancyLabel in zip(labels, units, fancyLabels):
    x = bip15.scFactorMdl.factors_[label]['home_team']['(Intercept)'] + \
        bip15.scFactorMdl.factors_[label]['home_team']['scImputedFALSE']
    y = bip16.scFactorMdl.factors_[label]['home_team']['(Intercept)'] + \
        bip16.scFactorMdl.factors_[label]['home_team']['scImputedFALSE']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, alpha=0)

    axLims = list(ax.axis())
    axLims[0] = axLims[2] = min(axLims[0::2])
    axLims[1] = axLims[3] = max(axLims[1::2])
    ax.axis(axLims)

    plotMLBLogos(x, y, ax=ax)

    ax.plot(axLims[:2], axLims[2:],
            '--', color=plt.rcParams['lines.color'], linewidth=1)

    ax.set_title('{} ({}) Venue Bias'.format(fancyLabel, unit))
    ax.set_xlabel('2015 Season')
    ax.set_ylabel('2016 Season')

    r2 = stats.pearsonr(x, y)[0] ** 2
    labels = ['R2: {:.2f}'.format(r2)]
    addText(ax, labels, loc='lower right')

    fig.savefig('{} 2015-2016 Correlation'.format(fancyLabel))

    x = bip15.scFactorMdl.factors_[label]['home_team']['(Intercept)'] + \
        bip15.scFactorMdl.factors_[label]['home_team']['scImputedTRUE']
    y = bip16.scFactorMdl.factors_[label]['home_team']['(Intercept)'] + \
        bip16.scFactorMdl.factors_[label]['home_team']['scImputedTRUE']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, alpha=0)

    axLims = list(ax.axis())
    axLims[0] = axLims[2] = min(axLims[0::2])
    axLims[1] = axLims[3] = max(axLims[1::2])
    ax.axis(axLims)

    plotMLBLogos(x, y, ax=ax)

    ax.plot(axLims[:2], axLims[2:],
            '--', color=plt.rcParams['lines.color'], linewidth=1)

    ax.set_title('Missing {} ({}) Venue Bias'.format(fancyLabel, unit))
    ax.set_xlabel('2015 Season')
    ax.set_ylabel('2016 Season')

    r2 = stats.pearsonr(x, y)[0] ** 2
    labels = ['R2: {:.2f}'.format(r2)]
    addText(ax, labels, loc='lower right')

    fig.savefig('Missing {} 2015-2016 Correlation'.format(fancyLabel))
