# %% Imports

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis

from statcast.bip import Bip
from statcast.better.spark import cross_val_predict
from statcast.tools.plot import plotPrecRec, plotPrecRecMN, plotResiduals


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

X1 = subData.loc[:, xLabels[:-1]]
X2 = subData.loc[:, xLabels]
y1 = subData[yLabel] == 'Home Run'
y2 = subData[yLabel]

skf = StratifiedKFold(n_splits=10)

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

y11pl = cross_val_predict(lda, X1, y1, cv=skf, n_jobs=-1,
                          method='predict_proba')
y21pl = cross_val_predict(lda, X2, y1, cv=skf, n_jobs=-1,
                          method='predict_proba')
y12pl = cross_val_predict(lda, X1, y2, cv=skf, n_jobs=-1,
                          method='predict_proba')
y22pl = cross_val_predict(lda, X2, y2, cv=skf, n_jobs=-1,
                          method='predict_proba')

y11pq = cross_val_predict(qda, X1, y1, cv=skf, n_jobs=-1,
                          method='predict_proba')
y21pq = cross_val_predict(qda, X2, y1, cv=skf, n_jobs=-1,
                          method='predict_proba')
y12pq = cross_val_predict(qda, X1, y2, cv=skf, n_jobs=-1,
                          method='predict_proba')
y22pq = cross_val_predict(qda, X2, y2, cv=skf, n_jobs=-1,
                          method='predict_proba')

y11pl = y11pl[:, 1]
y21pl = y21pl[:, 1]
y11pq = y11pq[:, 1]
y21pq = y21pq[:, 1]

y12pl = y12pl[:, [0, 1, 3, 4, 2]]
y22pl = y22pl[:, [0, 1, 3, 4, 2]]
y12pq = y12pq[:, [0, 1, 3, 4, 2]]
y22pq = y22pq[:, [0, 1, 3, 4, 2]]

# %% Log-loss

logL11l = log_loss(y1, y11pl)
logL21l = log_loss(y1, y21pl)
logL12l = log_loss(y2.cat.codes.values, y12pl)
logL22l = log_loss(y2.cat.codes.values, y22pl)

logL11q = log_loss(y1, y11pq)
logL21q = log_loss(y1, y21pq)
logL12q = log_loss(y2.cat.codes.values, y12pq)
logL22q = log_loss(y2.cat.codes.values, y22pq)

# %% Plot Precision-Recall Curve

fig = plotPrecRec(y1, y11pl, label='EV + LA: LL={:.2f}'.format(logL11l))
ax = fig.gca()
plotPrecRec(y1, y21pl, ax=ax, label='EV + LA + SA: LL={:.2f}'.format(logL21l))
ax.legend()
ax.set_title('LDA Homerun Classifier')
fig.savefig('LDA HR Prec-Rec Curve')

fig = plotPrecRecMN(y2, y12pl)
fig.gca().set_title('LDA(EV + LA): LL={:.2f}'.format(logL12l))
fig.savefig('LDA(EV + LA) Hit Prec-Rec Curves')
fig = plotPrecRecMN(y2, y22pl)
fig.gca().set_title('LDA(EV + LA + SA): LL={:.2f}'.format(logL22l))
fig.savefig('LDA(EV + LA + SA) Hit Prec-Rec Curves')

fig = plotPrecRec(y1, y11pq, label='EV + LA: LL={:.2f}'.format(logL11q))
ax = fig.gca()
plotPrecRec(y1, y21pq, ax=ax, label='EV + LA + SA: LL={:.2f}'.format(logL21q))
ax.legend()
ax.set_title('QDA Homerun Classifier')
fig.savefig('QDA HR Prec-Rec Curve')

fig = plotPrecRecMN(y2, y12pq)
fig.gca().set_title('QDA(EV + LA): LL={:.2f}'.format(logL12q))
fig.savefig('QDA(EV + LA) Hit Prec-Rec Curves')
fig = plotPrecRecMN(y2, y22pq)
fig.gca().set_title('QDA(EV + LA + SA): LL={:.2f}'.format(logL22q))
fig.savefig('QDA(EV + LA + SA) Hit Prec-Rec Curves')

# %% Plot Residuals

figs11l = plotResiduals(X1.values, y1 * 100, y11pl * 100,
                        xLabels=fancyLabels[:-1], xUnits=units[:-1],
                        yLabels=['Home Run'], yUnits=['%'],
                        pltParams={'ms': 1})
for label, fig in zip(xLabels[:-1], figs11l):
    fig.gca().set_title('LDA(EV + LA) HR Classifier')
    fig.savefig('LDA(EV + LA) HR Residuals over {}'.format(label))
figs21l = plotResiduals(X2.values, y1 * 100, y21pl * 100,
                        xLabels=fancyLabels, xUnits=units,
                        yLabels=['Home Run'], yUnits=['%'],
                        pltParams={'ms': 1})
for label, fig in zip(xLabels, figs21l):
    fig.gca().set_title('LDA(EV + LA + SA) HR Classifier')
    fig.savefig('LDA(EV + LA + SA) HR Residuals over {}'.format(label))

Y2 = np.zeros(shape=(y2.shape[0], len(y2.cat.categories)))
for y, code in zip(Y2, y2.cat.codes):
    y[code] = 1

figs12l = plotResiduals(X1.values, Y2 * 100, y12pl * 100,
                        xLabels=fancyLabels[:-1], xUnits=units[:-1],
                        yLabels=y2.cat.categories, yUnits=['%'] * Y2.shape[1],
                        pltParams={'ms': 1})
for label, fig in zip(xLabels[:-1], figs12l):
    fig.get_axes()[0].set_title('LDA(EV + LA) Hit Classifier')
    fig.savefig('LDA(EV + LA) Hit Residuals over {}'.format(label))
figs22l = plotResiduals(X2.values, Y2 * 100, y22pl * 100,
                        xLabels=fancyLabels, xUnits=units,
                        yLabels=y2.cat.categories, yUnits=['%'] * Y2.shape[1],
                        pltParams={'ms': 1})
for label, fig in zip(xLabels, figs22l):
    fig.get_axes()[0].set_title('LDA(EV + LA + SA) Hit Classifier')
    fig.savefig('LDA(EV + LA + SA) Hit Residuals over {}'.format(label))

figs11q = plotResiduals(X1.values, y1 * 100, y11pq * 100,
                        xLabels=fancyLabels[:-1], xUnits=units[:-1],
                        yLabels=['Home Run'], yUnits=['%'],
                        pltParams={'ms': 1})
for label, fig in zip(xLabels[:-1], figs11q):
    fig.gca().set_title('QDA(EV + LA) HR Classifier')
    fig.savefig('QDA(EV + LA) HR Residuals over {}'.format(label))
figs21q = plotResiduals(X2.values, y1 * 100, y21pq * 100,
                        xLabels=fancyLabels, xUnits=units,
                        yLabels=['Home Run'], yUnits=['%'],
                        pltParams={'ms': 1})
for label, fig in zip(xLabels, figs21q):
    fig.gca().set_title('QDA(EV + LA + SA) HR Classifier')
    fig.savefig('QDA(EV + LA + SA) HR Residuals over {}'.format(label))

figs12q = plotResiduals(X1.values, Y2 * 100, y12pq * 100,
                        xLabels=fancyLabels[:-1], xUnits=units[:-1],
                        yLabels=y2.cat.categories, yUnits=['%'] * Y2.shape[1],
                        pltParams={'ms': 1})
for label, fig in zip(xLabels[:-1], figs12q):
    fig.get_axes()[0].set_title('QDA(EV + LA) Hit Classifier')
    fig.savefig('QDA(EV + LA) Hit Residuals over {}'.format(label))
figs22q = plotResiduals(X2.values, Y2 * 100, y22pq * 100,
                        xLabels=fancyLabels, xUnits=units,
                        yLabels=y2.cat.categories, yUnits=['%'] * Y2.shape[1],
                        pltParams={'ms': 1})
for label, fig in zip(xLabels, figs22q):
    fig.get_axes()[0].set_title('QDA(EV + LA + SA) Hit Classifier')
    fig.savefig('QDA(EV + LA + SA) Hit Residuals over {}'.format(label))
