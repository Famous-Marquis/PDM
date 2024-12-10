"""
计算模块,提供了一些计算函数:
1. 直角坐标系与极坐标变换
2. 傅里叶变换
"""

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)


def meshgrid(x):
    x, y = x, x
    return np.meshgrid(x, y)


def ft2(g, delta):
    return fftshift(fft2(fftshift(g))) * delta**2


def ift2(G, delta_f):
    N = G.shape[0]
    return ifftshift(ifft2(ifftshift(G))) * (N * delta_f) ** 2


def circ(x, y, D):
    r = np.sqrt(x**2 + y**2)
    z = (r < D / 2).astype(float)
    z[r == 0.5] = 0.5
    return z


def randn(d):
    return np.random.randn(d, d)
