import numpy as np
from numpy import pi


def modified_Von_Karman(l0: float, L0: float):
    try:
        kappa_m = 5.92 / l0
    except ZeroDivisionError:
        kappa_m = float("inf")

    if L0 == float("inf"):
        kappa_0 = 0
    else:
        kappa_0 = 1 / L0

    def fun(kappa, Cn2: float = 2 * pi * 1e-16):
        if kappa_m == float("inf"):
            kappa_ratio = 0
        else:
            kappa_ratio = kappa / kappa_m
        PSD_phi = (
            0.033
            * Cn2
            * np.exp(-((kappa_ratio) ** 2))
            / ((kappa**2 + kappa_0**2) ** (11 / 6))
        )
        return PSD_phi

    return fun


def Kolmogorov():
    l0 = 0.0
    L0 = float("inf")
    return modified_Von_Karman(l0, L0)


def Von_Karman(L0):
    l0 = 0.0
    return modified_Von_Karman(l0, L0)
