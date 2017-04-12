import numpy as np


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
