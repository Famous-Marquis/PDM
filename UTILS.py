import numpy as np

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift


def ft2(g, dx):
    G = fftshift(fft2(fftshift(g))) * dx**2
    return G


def ift2(G, df):
    N = G.shape[0]
    g = ifftshift(ifft2(ifftshift(G))) * (N * df) ** 2
    return g
