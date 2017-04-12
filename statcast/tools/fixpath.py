import sys
import os
from pathlib import Path


_basePath = sys.path.copy()


def resetPath():
    '''Doc String'''

    sys.path = _basePath.copy()


def findFile(fileName, searchDirs=sys.path, subs=False,
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
                filePaths.extend(findFile(fileName,
                                          searchPaths=absSubDirs,
                                          subs=subs,
                                          findAll=findAll))
    if findAll:
        return filePaths
    raise FileNotFoundError('Could not find {}'.format(fileName))


def addPath(path=Path.cwd(), subs=True, reset=False):
        '''Doc String'''

        if reset:
            resetPath()
        if isinstance(path, (tuple, list)):
            for p in reversed(path):
                addPath(path=p, subs=subs)
            return

        path = str(path)
        if path in sys.path:
            sys.path.remove(path)

        if subs:
            for child in Path(path).iterdir():
                if child.is_dir():
                    addPath(child, subs=subs)
        sys.path.insert(0, str(path))
