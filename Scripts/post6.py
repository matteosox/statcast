#%%

%reset -f

#%% Create data for regression from 2016 regular season

import bbsavant
import pandas as pd
import numpy as np

db = bbsavant.db()

rawD = pd.read_sql_query(
    '''SELECT events, hit_speed, hit_angle, hit_distance_sc, hc_x, hc_y
    FROM {}
    WHERE hit_speed IS NOT NULL
    AND hit_angle IS NOT NULL
    AND hit_distance_sc IS NOT NULL
    AND hc_x IS NOT NULL
    AND hc_y IS NOT NULL
    AND game_year IS 2016
    AND game_type IS 'R ' '''.format(db._tblName),db.engine)

rawD['sa'] = (np.arctan2(208 - rawD.hc_y,rawD.hc_x - 128) / (2 * np.pi) * 360 + 90) % 360 - 180
rawD = rawD[(rawD.events != 'Batter Interference')
            & (rawD.events != 'Hit By Pitch')
            & (rawD.events != 'Strikeout')
            & (rawD.events != 'Walk')]

#%% Plot histogram over baseball diamond

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

xs = rawD.hit_distance_sc * np.cos((rawD.sa + 90) / 360 * 2 * np.pi)
ys = rawD.hit_distance_sc * np.sin((rawD.sa + 90) / 360 * 2 * np.pi)

fll = rawD.hit_distance_sc.max() / np.sqrt(2)

fig = plt.figure() # generate figure
ax = fig.add_subplot(1, 1, 1) # add subplot to figure, 1 row, 1 col, #1 active
h2d = ax.hist2d(xs, ys, bins=100, norm=LogNorm())
plt.colorbar(h2d[3])
ax.hold(True)
ax.plot((0, fll), (0, fll),'k',
        (0, -fll), (0, fll),'k')
ax.axis('equal')

#%% GLM for home run likelihood

from statsmodels import api as sm

endog = rawD.events == 'Home Run'
exog = rawD.loc[:,['hit_speed', 'hit_angle', 'sa']]

mdl = sm.GLM(endog, sm.tools.tools.add_constant(exog), family=sm.families.Binomial(sm.families.links.probit))
results = mdl.fit()
print(results.summary())
print(results.summary2())

fig = plt.figure(figsize=(10,10),tight_layout=True)

axs = []

for ii in range(exog.shape[1]):
    ax = fig.add_subplot(exog.shape[1],1,ii + 1)
    ax.plot(exog.iloc[:,ii],results.resid_response,'.')
    ax.set_xlabel(exog.columns[ii], family='Helvetica Neue')
    ax.set_ylabel('Residuals', family='Helvetica Neue')
    axs.append(ax)

axs[0].set_title('Home Run Likelihood', family='Helvetica Neue')

#%% Multinomial Logit for Out, Single, Double, Triple, Homerun likelihood

rawD2 = rawD

rawD2.events[rawD2.events == 'Bunt Groundout'] = 'Out'
rawD2.events[rawD2.events == 'Double Play'] = 'Out'
rawD2 = rawD2[rawD2.events != 'Fan interference']
rawD2 = rawD2[rawD2.events != 'Field Error']
rawD2.events[rawD2.events == 'Fielders Choice'] = 'Out'
rawD2.events[rawD2.events == 'Fielders Choice Out'] = 'Out'
rawD2.events[rawD2.events == 'Flyout'] = 'Out'
rawD2.events[rawD2.events == 'Forceout'] = 'Out'
rawD2.events[rawD2.events == 'Grounded Into DP'] = 'Out'
rawD2.events[rawD2.events == 'Groundout'] = 'Out'
rawD2.events[rawD2.events == 'Lineout'] = 'Out'
rawD2.events[rawD2.events == 'Pop Out'] = 'Out'
rawD2.events[rawD2.events == 'Runner Out'] = 'Out'
rawD2.events[rawD2.events == 'Sac Bunt'] = 'Out'
rawD2.events[rawD2.events == 'Sac Fly'] = 'Out'
rawD2.events[rawD2.events == 'Sac Fly DP'] = 'Out'
rawD2.events[rawD2.events == 'Triple Play'] = 'Out'
endog = rawD2.events.astype('category')

exog = rawD2.loc[:, ['hit_speed', 'hit_angle', 'sa']]

mdl2 = sm.MNLogit(endog, sm.tools.tools.add_constant(exog))
results2 = mdl2.fit()
print(results2.summary())
# print(results2.summary2())

for ii in range(len(endog.cat.categories)):
    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    axs = []
    for jj in range(exog.shape[1]):
        ax = fig.add_subplot(exog.shape[1], 1, jj + 1)
        ax.plot(exog.iloc[:, jj], (endog == endog.cat.categories[ii]) - results2.predict()[:, ii], '.')
        ax.set_xlabel(exog.columns[jj],family='Helvetica')
        ax.set_ylabel('Residuals')
        axs.append(ax)
    axs[0].set_title('{} Likelihood'.format(endog.cat.categories[ii]))
