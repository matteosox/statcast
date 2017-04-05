import numpy as np
from pathlib import Path
import sys
import os


def conv(f, xf, g, xg, mode='full', dx=None):
    '''Doc String'''

    dxf, dxg = np.diff(xf[:2]), np.diff(xg[:2])
    if not dx:
        if dxf == dxg:
            dx = dxf
        else:
            dx = dxf * dxg

    xfr = np.arange(xf[0], xf[-1] + dx / 2, dx)
    xgr = np.arange(xg[0], xg[-1] + dx / 2, dx)

    fr = np.interp(xfr, xf, f, right=0)
    gr = np.interp(xgr, xg, g, right=0)

    h = np.convolve(fr, gr, mode=mode) * dx
    xh = np.arange(xfr[0] + xgr[0], xfr[-1] + xgr[-1] + dx / 2, dx)
    if mode == 'same':
        skip = min(len(xfr), len(xgr)) // 2
        l = max(len(xfr), len(xgr))
        xh = xh[skip:(skip + l)]
    elif mode == 'valid':
        skip = min(len(xfr), len(xgr)) - 1
        xh = xh[skip:(-1 - skip)]
    return h, xh


class fixPath():
    '''Doc String'''

    _basePath = sys.path.copy()

    def __init__(self, *args, **kwargs):
        '''Doc String'''

        self.addPath(*args, **kwargs)
        return

    def addPath(self, path=Path.cwd(), subs=True, reset=False):
        '''Doc String'''

        if reset:
            self.resetPath()
        if isinstance(path, (tuple, list)):
            for p in reversed(path):
                self.addPath(path=p, subs=subs)
            return

        path = str(path)
        if path in sys.path:
            sys.path.remove(path)

        if subs:
            for child in Path(path).iterdir():
                if child.is_dir():
                    self.addPath(child, subs=subs)
        sys.path.insert(0, str(path))
        return

    @classmethod
    def resetPath(cls):
        '''Doc String'''

        sys.path = cls._basePath.copy()
        return

    @classmethod
    def findFile(cls, fileName, searchDirs=sys.path, subs=False,
                 findAll=False):
        '''Doc String'''

        if searchDirs is None:
            searchDirs = [Path.cwd().anchor]
            subs = True
        elif isinstance(searchDirs, str):
            searchDirs = [searchDirs]
        filePaths = []
        for searchDir in searchDirs:
            for root, subDirs, files in os.walk(os.path.abspath(searchDir)):
                if fileName in files:
                    filePath = os.path.join(root, fileName)
                    if not findAll:
                        return filePath
                    filePaths.append(filePath)
                if subs:
                    absSubDirs = [os.path.join(root, subDir)
                                  for subDir in subDirs]
                    filePaths.extend(cls.findFile(fileName,
                                                  searchPaths=absSubDirs,
                                                  subs=subs,
                                                  findAll=findAll))
        if findAll:
            return filePaths
        return ''
