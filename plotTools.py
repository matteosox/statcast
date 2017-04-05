import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib import lines as mlines
from betterModels import betterKernelDensity
from sklearn.model_selection import GridSearchCV
from PIL import Image as pilimg


def correlationPlot(Y, Yp, labels=None, units=None, **plotParams):
    '''Doc String'''

    plt.style.use('personal')

    # Handle Pandas DataFrames
    if isinstance(Y, pd.DataFrame):
        if labels is None:
            labels = list(Y.columns)
        Y = Y.values
    if isinstance(Yp, pd.DataFrame):
        Yp = Yp.values

    # Handle 1D arrays
    if Y.ndim == 1:
        Y = Y[:, None]
    if Yp.ndim == 1:
        Yp = Yp[:, None]

    # Handle row vectors
    if Y.shape[0] == 1:
        Y = Y.T
    if Yp.shape[0] == 1:
        Yp = Yp.T

    # Handle no label or unit inputs
    if labels is None:
        labels = [None for dummy in range(Y.shape[1])]
    if units is None:
        units = [None for dummy in range(Y.shape[1])]

    figs = []

    for y, yp, label, unit in zip(Y.T, Yp.T,
                                  labels, units):
        rmsErr = np.sqrt(np.mean((y - yp) ** 2))
        r2 = stats.pearsonr(y, yp)[0] ** 2
        mae = np.mean(np.abs(y - yp))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(y, yp, '.', **plotParams)

        if unit is not None:
            ax.set_xlabel('Actual ({})'.format(unit))
            ax.set_ylabel('Prediction ({})'.format(unit))
        else:
            ax.set_xlabel('Actual')
            ax.set_ylabel('Prediction')

        if label is not None:
            ax.set_title(label)

        axLims = list(ax.axis())
        axLims[0] = axLims[2] = min(axLims[0::2])
        axLims[1] = axLims[3] = max(axLims[1::2])
        ax.axis(axLims)
        ax.plot(axLims[:2], axLims[2:], '--', color='0.1', linewidth=1)

        labels = ['{}: {:.2f}'.format(name, stat)
                  for name, stat in zip(['RMSE', 'R2', 'MAE'],
                                        [rmsErr, r2, mae])]
        addText(ax, labels, loc='lower right')

        figs.append(fig)

    return figs


def addText(ax, text, loc='best', **kwargs):
    '''Doc String'''

    if 'right' in loc:
        markerfirst = False
    else:
        markerfirst = True

    handles = [mlines.Line2D([], [], alpha=0.0)] * len(text)
    ax.legend(handles=handles, labels=text, loc=loc, frameon=False,
              handlelength=0, handletextpad=0, markerfirst=markerfirst,
              **kwargs)
    return


def plotKDHist(data,
               kernel='epanechnikov', bandwidth=None, alpha=5e-2, ax=None):
    '''Doc String'''

    if data.ndim < 2:
        fitData = data[:, None]
    else:
        fitData = data
    xmin, xmax = min(data), max(data)
    xFit = np.linspace(xmin, xmax, 1e3)

    if bandwidth is None:
        kde = betterKernelDensity(kernel=kernel, rtol=1e-4)
        parameters = {'bandwidth': np.logspace(-3, -1, num=10) * (xmax - xmin)}
        trainGrid = GridSearchCV(kde, parameters, cv=3, n_jobs=-1).fit(fitData)
        kde = trainGrid.best_estimator_
    else:
        kde = betterKernelDensity(kernel=kernel, rtol=1e-4,
                                  bandwidth=bandwidth).fit(fitData)

    fitL, fitU = kde.confidence(xFit[:, None], alpha=alpha)

    if ax is not None:
        ax.fill_between(xFit, fitU, fitL, alpha=0.35, lw=0,
                        label='{:.0f}% Confidence Interval'.
                        format(1e2 * (1 - alpha)))
        return ax, kde

    plt.style.use('personal')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.fill_between(xFit, fitU, fitL, alpha=0.35, lw=0,
                    label='{:.0f}% Confidence Interval'.
                    format(1e2 * (1 - alpha)))

    try:
        ax.set_xlabel(data.name)
    except:
        pass

    ax.set_ylabel('Probability Density (%)')
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=0)
    return fig, kde


def plotImages(X, Y, images, sizes=20, alphas=1, ax=None):
    '''Doc String'''

    if not isinstance(sizes, (list, tuple)):
        sizes = (sizes,) * len(X)
    if not isinstance(images, (list, tuple)):
        images = (images,) * len(X)
    if not isinstance(alphas, (list, tuple)):
        alphas = (alphas,) * len(X)

    ims = [pilimg.open(image) for image in images]

    if ax is None:
        plt.style.use('personal')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # plot and set axis limits
    ax.plot(X, Y, 'o', mfc='None', mec='None', markersize=max(sizes))
    ax.axis(ax.axis())

    for size, im, alpha, x, y in zip(sizes, ims, alphas, X, Y):
        offsetsPx = np.array([sz / max(im.size) * size / 72 / 2 *
                              ax.get_figure().dpi for sz in im.size])
        pxPerUnit = ax.transData.transform((1, 1)) - \
            ax.transData.transform((0, 0))
        offsetsUnit = offsetsPx / pxPerUnit
        extent = (x - offsetsUnit[0], x + offsetsUnit[0],
                  y - offsetsUnit[1], y + offsetsUnit[1])
        ax.imshow(im, alpha=alpha, extent=extent, aspect='auto',
                  interpolation='bilinear')

    try:
        return fig
    except:
        return ax
