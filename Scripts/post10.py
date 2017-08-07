# %% Imports

import os
import datetime

import requests
from pyspark import SparkContext

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

from statcast.bip import Bip
from statcast.better.spark import cross_val_predict
from statcast.tools.plot import plotPrecRec, plotPrecRecMN, plotResiduals
from statcast.better.declassifier import KDEClassifier


# %% Create Spark Context

sc = SparkContext(appName='post10')

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

skf = StratifiedKFold(n_splits=10, shuffle=True)

kdc = KDEClassifier(kdeParams=dict(kernel='gaussian'),
                    n_jobs=-1)

y11p = cross_val_predict(kdc, X1, y1, cv=skf, n_jobs=sc,
                         method='predict_proba')
y21p = cross_val_predict(kdc, X2, y1, cv=skf, n_jobs=sc,
                         method='predict_proba')
y12p = cross_val_predict(kdc, X1, y2, cv=skf, n_jobs=sc,
                         method='predict_proba')
y22p = cross_val_predict(kdc, X2, y2, cv=skf, n_jobs=sc,
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
fig.savefig('KDC HR Prec-Rec Curve')

fig = plotPrecRecMN(y2, y12p)
fig.gca().set_title('EV + LA: LL={:.2f}'.format(logL12))
fig.savefig('KDC(EV + LA) Hit Prec-Rec Curves')
fig = plotPrecRecMN(y2, y22p)
fig.gca().set_title('EV + LA + SA: LL={:.2f}'.format(logL22))
fig.savefig('KDC(EV + LA + SA) Hit Prec-Rec Curves')

# %% Plot Residuals

figs11 = plotResiduals(X1.values, y1 * 100, y11p * 100,
                       xLabels=fancyLabels[:-1], xUnits=units[:-1],
                       yLabels=['Home Run'], yUnits=['%'], pltParams={'ms': 1})
for label, fig in zip(xLabels[:-1], figs11):
    fig.gca().set_title('HR(EV + LA) Classifier')
    fig.savefig('KDC(EV + LA) HR Residuals over {}'.format(label))
figs21 = plotResiduals(X2.values, y1 * 100, y21p * 100,
                       xLabels=fancyLabels, xUnits=units,
                       yLabels=['Home Run'], yUnits=['%'], pltParams={'ms': 1})
for label, fig in zip(xLabels, figs21):
    fig.gca().set_title('HR(EV + LA + SA) Classifier')
    fig.savefig('KDC(EV + LA + SA) HR Residuals over {}'.format(label))

Y2 = LabelBinarizer().fit_transform(y2)
y2Labels = sorted(y2.cat.categories)

figs12 = plotResiduals(X1.values, Y2 * 100, y12p * 100,
                       xLabels=fancyLabels[:-1], xUnits=units[:-1],
                       yLabels=y2Labels, yUnits=['%'] * Y2.shape[1],
                       pltParams={'ms': 1})
for label, fig in zip(xLabels[:-1], figs12):
    fig.get_axes()[0].set_title('Hit(EV + LA) Classifier')
    fig.savefig('KDC(EV + LA) Hit Residuals over {}'.format(label))
figs22 = plotResiduals(X2.values, Y2 * 100, y22p * 100,
                       xLabels=fancyLabels, xUnits=units,
                       yLabels=y2Labels, yUnits=['%'] * Y2.shape[1],
                       pltParams={'ms': 1})
for label, fig in zip(xLabels, figs22):
    fig.get_axes()[0].set_title('Hit(EV + LA + SA) Classifier')
    fig.savefig('KDC(EV + LA + SA) Hit Residuals over {}'.format(label))

# %% Transfer results to S3

instanceID = requests. \
        get('http://169.254.169.254/latest/meta-data/instance-id').text
dtStr = datetime.datetime.utcnow().strftime('%Y-%m-%d--%H-%M-%S')
os.system('aws s3 sync . s3://mf-first-bucket/output/{}/{}'.
          format(instanceID, dtStr))

# %% Stop Spark Context

sc.stop()
