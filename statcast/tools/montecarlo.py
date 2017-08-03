from copy import deepcopy
import numpy as np

eps = np.sqrt(np.spacing(1))


def rectSampler(lims):
    '''Doc String'''

    x0 = np.array([lim[0] for lim in lims])
    d = np.array([lim[1] - lim[0] for lim in lims])
    ndim = x0.size
    vol = d.prod()

    return lambda n: (
        np.random.rand(n, ndim) * np.tile(d, (n, 1)) + np.tile(x0, (n, 1))
        ), vol


def integrate(f, sampler, vol, relErr=0, absErr=eps, n0=100):
    '''Doc String'''

    X = sampler(n0)
    y = f(X)

    while True:
        mu = y.mean()
        std = np.std(y, ddof=1)
        i = vol * mu
        se = vol * std / np.sqrt(y.size)
        if se <= absErr:
            break
        if i != 0:
            if np.abs(se / i) <= relErr:
                break

        nA = (vol * std / absErr) ** 2
        if relErr > 0:
            nR = (std / relErr / mu) ** 2
        else:
            nR = nA + 1
        n = np.ceil(min(nA, nR) - y.size).astype(int)
        X = sampler(n)
        y = np.concatenate([y, f(X)])

    return i, se


class Region():
    '''Doc String'''

    def __init__(self, f, lims, X=None, y=None):
        '''Doc String'''

        self.X = X
        self.y = y

        self.lims = lims
        self.f = f
        self.sampler, self.vol = rectSampler(lims)

    def sample(self, n):
        '''Doc String'''

        if n == 0:
            return
        X = self.sampler(n)
        y = self.f(X)

        if self.X is not None:
            self.X = np.concatenate([self.X, X])
        else:
            self.X = X

        if self.y is not None:
            self.y = np.concatenate([self.y, y])
        else:
            self.y = y

    @property
    def integral(self):
        return self.y.mean() * self.vol

    def se(self, n=None):
        if n is None:
            n = self.y.size
        return self.vol * self.y.std(ddof=1) / np.sqrt(n)

    def seImprovement(self, n):
        return self.se() - self.se(self.y.size + n)

    def split(self, n):
        '''Doc String'''

        variance = np.inf
        nOld = self.y.size
        nNew = n + nOld

        for i, (xMin, xMax) in enumerate(self.lims):
            xMid = (xMin + xMax) / 2

            inds = self.X[:, i] <= xMid
            lStd = self.y[inds].std(ddof=1)
            rStd = self.y[~inds].std(ddof=1)
            lNOld = inds.sum()
            rNOld = nOld - lNOld
            lNNew = min(max(np.round(lStd / (lStd + rStd) * nNew).astype(int),
                        lNOld), lNOld + n)
            rNNew = nNew - lNNew
            newVar = lStd ** 2 / (4 * lNNew) + rStd ** 2 / (4 * rNNew)
            if newVar < variance:
                variance = newVar
                splitInd = i
                lNAdd = lNNew - lNOld
                rNAdd = rNNew - rNOld

        lLims = deepcopy(self.lims)
        rLims = deepcopy(self.lims)
        xMid = sum(self.lims[splitInd]) / 2
        lLims[splitInd][1] = xMid
        rLims[splitInd][0] = xMid
        inds = self.X[:, splitInd] <= xMid
        lX = self.X[inds, :]
        ly = self.y[inds]
        rX = self.X[~inds, :]
        ry = self.y[~inds]

        lRegion = Region(self.f, lLims, lX, ly)
        lRegion.sample(lNAdd)
        rRegion = Region(self.f, rLims, rX, ry)
        rRegion.sample(rNAdd)

        return lRegion, rRegion


def stratifiedIntegrate(f, lims, relErr=0, absErr=eps, n=100):
    '''Doc String'''

    region = Region(f, lims)
    region.sample(4 * n)
    regions = [region]
    se2s = [region.se() ** 2 for region in regions]
    ints = [region.integral for region in regions]
    seImps = [region.seImprovement(2 * n) for region in regions]

    while True:
        i = sum(ints)
        se = np.sqrt(sum(se2s))
        if se <= absErr:
            break
        if i != 0:
            if np.abs(se / i) <= relErr:
                break
        ind = np.argmax(seImps)
        region = regions.pop(ind)
        del se2s[ind]
        del ints[ind]
        del seImps[ind]
        newRegions = region.split(2 * n)
        regions.extend(newRegions)
        se2s.extend([newRegion.se() ** 2 for newRegion in newRegions])
        ints.extend([newRegion.integral for newRegion in newRegions])
        seImps.extend([newRegion.seImprovement(2 * n)
                       for newRegion in newRegions])

    return i, se
