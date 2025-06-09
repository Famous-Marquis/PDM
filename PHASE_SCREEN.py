import time

import tqdm

import UTILS
from numpy import pi
from numpy.random import default_rng
import numpy as np

rng = default_rng()


def vaccum(N):
    return np.ones([N, N])


def ft_phase_screen(N, dx, psd, Cn2):
    """
    参数
    -------------
    N:

    """
    df = 1 / (N * dx)
    fx = np.linspace(-N // 2, N // 2, N, endpoint=False) * df
    fx, fy = np.meshgrid(fx, fx)
    f = np.sqrt(fx**2 + fy**2)
    psd_phi = psd()(f, Cn2)

    psd_phi[N // 2, N // 2] = 0
    cn = (
        (rng.standard_normal([N, N]) + 1j * rng.standard_normal([N, N]))
        * np.sqrt(psd_phi)
        * df
    )
    phz = np.real(UTILS.ift2(cn, 1))
    return phz


def sh_phase_screen(N, dx, psd, Cn2, sub_harm):
    D = N * dx
    x = np.linspace(-D / 2, D / 2 - dx, N)
    x, y = np.meshgrid(x, x)

    phi_lo = np.zeros([N, N])
    for p in range(1, sub_harm + 1):
        df = 1 / (D * 5**p)
        fx = np.linspace(-1, 1, 3)
        fx, fy = np.meshgrid(fx, fx)
        f = np.sqrt(fx**2 + fy**2)
        psd_phi = psd()(f, Cn2)
        cn = complex(rng.normal(3), rng.normal(3)) * np.sqrt(psd_phi) * df
        sub_harmonics = np.zeros([N, N])
        for i in range(3):
            for j in range(3):
                sub_harmonics = cn[i, j] * np.exp(
                    1j * 2 * pi * (fx[i, j] * x + fy[i, j] * y)
                )
        phi_lo += np.real(sub_harmonics)
    phi_lo = np.real(phi_lo) - np.mean(phi_lo)
    return phi_lo


def ft_sh_phsse_screen(N: int, dx, psd, sub_harm: int, Cn2: float = 2 * pi * 1e-16):
    phi_hi = ft_phase_screen(N=N, dx=dx, psd=psd, Cn2=Cn2)
    phi_lo = sh_phase_screen(N=N, dx=dx, psd=psd, Cn2=Cn2, sub_harm=sub_harm)
    return phi_hi + phi_lo

def timing_per_screen(repeat_num,N,dx,psd,sub_harm):
    start_time=time.time()
    for i in tqdm.trange(repeat_num,desc='fft-sh generating...'):
        screen = ft_sh_phsse_screen(N=N,dx=dx,psd=psd,sub_harm=4)
    end_time = time.time()
    average_time = (end_time-start_time)/repeat_num
    print(f'N:{N},SH:{sub_harm}\naverage time:', average_time)
    return average_time