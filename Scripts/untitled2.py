# %% Imports

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.preprocessing import LabelBinarizer

from statcast.bip import Bip
from statcast.tools.plot import plotPrecRec, plotPrecRecMN, plotResiduals
from statcast.better.declassifier import KDEClassifier


# %%

bip = Bip(years=(2016,), n_jobs=-1)

# %%

xLabels = ['hit_speed', 'hit_angle', 'sprayAngle']
fancyLabels = ['Exit Velocity', 'Launch Angle', 'Spray Angle']
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

X1 = subData.loc[:, xLabels[:-1]].values
X2 = subData.loc[:, xLabels].values
y1 = (subData[yLabel] == 'Home Run').values
y2 = subData[yLabel].values

skf = StratifiedKFold(n_splits=10, shuffle=True)
test, train = next(skf.split(X1, y1))

kdc = KDEClassifier(kdeParams=dict(kernel='gaussian'),
                    n_jobs=-1)
est11 = clone(kdc).fit(X1[train], y1[train])
est12 = clone(kdc).fit(X1[train], y2[train])
est21 = clone(kdc).fit(X2[train], y1[train])
est22 = clone(kdc).fit(X2[train], y2[train])

y11p = est11.predict_proba(X1[test])
y12p = est12.predict_proba(X1[test])
y21p = est21.predict_proba(X2[test])
y22p = est22.predict_proba(X2[test])

y11p = y11p[:, 1]
y21p = y21p[:, 1]

# %% Log-loss

logL11 = log_loss(y1[test], y11p)
logL21 = log_loss(y1[test], y21p)
logL12 = log_loss(y2[test], y12p)
logL22 = log_loss(y2[test], y22p)

# %% Plot Precision-Recall Curve

fig = plotPrecRec(y1[test], y11p, label='EV + LA: LL={:.2f}'.format(logL11))
ax = fig.gca()
plotPrecRec(y1[test], y21p, ax=ax, label='EV + LA + SA: LL={:.2f}'.format(logL21))
ax.legend()
ax.set_title('KDC Homerun Classifier')
#fig.savefig('KDC HR Prec-Rec Curve')

fig = plotPrecRecMN(y2[test], y12p)
fig.gca().set_title('EV + LA: LL={:.2f}'.format(logL12))
#fig.savefig('KDC(EV + LA) Hit Prec-Rec Curves')
fig = plotPrecRecMN(y2[test], y22p)
fig.gca().set_title('EV + LA + SA: LL={:.2f}'.format(logL22))
#fig.savefig('KDC(EV + LA + SA) Hit Prec-Rec Curves')

# %% Plot Residuals

figs11 = plotResiduals(X1[test], y1[test] * 100, y11p * 100,
                       xLabels=fancyLabels[:-1], xUnits=units[:-1],
                       yLabels=['Home Run'], yUnits=['%'], pltParams={'ms': 1})
for label, fig in zip(xLabels[:-1], figs11):
    fig.gca().set_title('HR(EV + LA) Classifier')
#    fig.savefig('KDC(EV + LA) HR Residuals over {}'.format(label))
figs21 = plotResiduals(X2[test], y1[test] * 100, y21p * 100,
                       xLabels=fancyLabels, xUnits=units,
                       yLabels=['Home Run'], yUnits=['%'], pltParams={'ms': 1})
for label, fig in zip(xLabels, figs21):
    fig.gca().set_title('HR(EV + LA + SA) Classifier')
#    fig.savefig('KDC(EV + LA + SA) HR Residuals over {}'.format(label))

Y2 = LabelBinarizer().fit_transform(y2)

figs12 = plotResiduals(X1[test], Y2[test] * 100, y12p * 100,
                       xLabels=fancyLabels[:-1], xUnits=units[:-1],
                       yLabels=est12.classes_, yUnits=['%'] * Y2.shape[1],
                       pltParams={'ms': 1})
for label, fig in zip(xLabels[:-1], figs12):
    fig.get_axes()[0].set_title('Hit(EV + LA) Classifier')
#    fig.savefig('KDC(EV + LA) Hit Residuals over {}'.format(label))
figs22 = plotResiduals(X2[test], Y2[test] * 100, y22p * 100,
                       xLabels=fancyLabels, xUnits=units,
                       yLabels=est22.classes_, yUnits=['%'] * Y2.shape[1],
                       pltParams={'ms': 1})
for label, fig in zip(xLabels, figs22):
    fig.get_axes()[0].set_title('Hit(EV + LA + SA) Classifier')
#    fig.savefig('KDC(EV + LA + SA) Hit Residuals over {}'.format(label))
