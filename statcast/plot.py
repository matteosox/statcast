import os

from .tools.fixpath import findFile
from .tools.plot import plotImages

from . import __path__

_logoPath = os.path.join(__path__[0], 'Team Logos')


def plotMLBLogos(X, Y, sizes=20, alphas=1, ax=None):
    '''Doc String'''

    images = [findFile('{}.png'.format(team.strip()), searchDirs=_logoPath)
              for team in X.index]
    thing = plotImages(X, Y, images, sizes, alphas, ax)
    if ax is None:
        ax = thing.gca()
    ax.set_xlabel(X.name)
    ax.set_ylabel(Y.name)
    return thing
