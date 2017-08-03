# %% Imports

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from statcast.bip import Bip
from statcast.better.spark import cross_val_predict
from statcast.tools.plot import plotPrecRec, plotPrecRecMN, plotResiduals
from statcast.better.declassifier import KDEClassifier


# %%

bip = Bip(years=(2016,), n_jobs=-1)

# %%

xLabels = ['hit_speed', 'hit_angle', 'sprayAngle']
units = ['mph', 'degrees', 'degrees']
yLabel = 'events'

subData = bip.data.loc[~bip.data['exclude'], xLabels + [yLabel]]

outs = ['Bunt Groundout', 'Double Play', 'Fielders Choice',
        'Fielders Choice Out', 'Flyout', 'Forceout', 'Grounded Into DP',
        'Groundout', 'Lineout', 'Pop Out', 'Runner Out', 'Sac Bunt',
        'Sac Fly', 'Sac Fly DP', 'Triple Play', 'Bunt Pop Out', 'Bunt Lineout',
        'Sacrifice Bunt DP']

subData['events'] = subData['events'].cat.add_categories(['Out'])

for out in outs:
    subData.loc[subData['events'] == out, 'events'] = 'Out'

subData['events'] = subData['events'].cat.remove_unused_categories()

X1 = subData.loc[:, xLabels[:-1]]
X2 = subData.loc[:, xLabels]
y1 = subData[yLabel] == 'Home Run'
y2 = subData[yLabel]

skf = StratifiedKFold(n_splits=10, shuffle=True)

kdc = KDEClassifier(kdeParams=dict(kernel='epanechnikov', rtol=1e-4),
                    n_jobs=-1)

y11p = cross_val_predict(kdc, X1, y1, cv=skf, n_jobs=-1,
                         method='predict_proba')
y21p = cross_val_predict(kdc, X2, y1, cv=skf, n_jobs=-1,
                         method='predict_proba')
y12p = cross_val_predict(kdc, X1, y2, cv=skf, n_jobs=-1,
                         method='predict_proba')
y22p = cross_val_predict(kdc, X2, y2, cv=skf, n_jobs=-1,
                         method='predict_proba')

y11p = y11p[:, 1]
y21p = y21p[:, 1]

# %% Log-loss

logL11 = log_loss(y1, y11p)
logL21 = log_loss(y1, y21p)
logL12 = log_loss(y2, y12p)
logL22 = log_loss(y2, y22p)

# %% Plot Precision-Recall Curve

fig = plotPrecRec(y1, y11p, label='EV + LA: LL={:.2f}'.format(logL11))
ax = fig.gca()
plotPrecRec(y1, y21p, ax=ax, label='EV + LA + SA: LL={:.2f}'.format(logL21))
ax.legend()
ax.set_title('KDC Homerun Classifier')

fig = plotPrecRecMN(y2, y12p)
fig.gca().set_title('KDC(EV + LA): LL={:.2f}'.format(logL12))
fig = plotPrecRecMN(y2, y22p)
fig.gca().set_title('KDC(EV + LA + SA): LL={:.2f}'.format(logL22))

# %% Plot Residuals

#figs11 = plotResiduals(X1.values, y1 * 100, y11p * 100,
#                       xLabels=xLabels[:-1], xUnits=units[:-1],
#                       yLabels=['Home Run'], yUnits=['%'], pltParams={'ms': 1})
#figs21 = plotResiduals(X2.values, y1 * 100, y21p * 100,
#                       xLabels=xLabels, xUnits=units,
#                       yLabels=['Home Run'], yUnits=['%'], pltParams={'ms': 1})
#
#Y2 = np.zeros(shape=(y2.shape[0], len(y2.cat.categories)))
#for y, code in zip(Y2, y2.cat.codes):
#    y[code] = 1
#
#figs12 = plotResiduals(X1.values, Y2 * 100, y12p * 100,
#                       xLabels=xLabels[:-1], xUnits=units[:-1],
#                       yLabels=y2.cat.categories, yUnits=['%'] * Y2.shape[1],
#                       pltParams={'ms': 1})
#figs22 = plotResiduals(X2.values, Y2 * 100, y22p * 100,
#                       xLabels=xLabels, xUnits=units,
#                       yLabels=y2.cat.categories, yUnits=['%'] * Y2.shape[1],
#                       pltParams={'ms': 1})
