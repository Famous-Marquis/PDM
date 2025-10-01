import math
from math import gamma
import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt

def A_alpha(alpha=11 / 3):
    result = (
            2 ** (alpha - 6)
            * (alpha ** 2 - 5 * alpha + 6)
            * math.pi ** (-3 / 2)
            * gamma((alpha - 2) / 2)
            / gamma((5 - alpha) / 2)
    )
    return result


def modified_Von_Karman(l0: float, L0: float,r0,alpha):
    try:
        kappa_m = 5.92 / l0
    except ZeroDivisionError:
        kappa_m = float("inf")

    if L0 == float("inf"):
        kappa_0 = 0
    else:
        kappa_0 = 1 / L0

    def fun(kappa,is_mp=False):
        if kappa_m == float("inf"):
            kappa_ratio = 0
        else:
            kappa_ratio = kappa / kappa_m
#! Warning: mp and np are not compatible in different implementations. When error occurs, please change another.
        PSD_phi = (
                1
                #2*mp.pi
                / 0.432
                # 0.5
                * A_alpha(alpha)
                * r0 ** (-5 / 3)
                * np.exp(-((kappa_ratio) ** 2))
                / ((kappa ** 2 + kappa_0 ** 2) ** (alpha / 2))
        )


        return PSD_phi

    return fun


def Kolmogorov(r0=0.05):
    l0 = 0.0
    L0 = float("inf")
    alpha=11/3
    return modified_Von_Karman(l0, L0,r0,alpha)


def Von_Karman(L0,r0,alpha=11/3):
    l0 = 0.0
    return modified_Von_Karman(l0, L0,r0,alpha)


def cov_j_j_prime(n, nn, m, mm, Phi,R):
    def integrand(kappa):
        factor = 2 * mp.pi * R * kappa
        Jn = mp.besselj(n + 1, factor)/factor if kappa!=0 else mp.mpf(1)
        Jn_prime = mp.besselj(nn + 1, factor) / factor if kappa != 0 else mp.mpf(1)
        return kappa*Jn*Jn_prime*Phi(kappa)
    integration = mp.quad(integrand,[0,mp.inf])
    result = 2*mp.pi*(-1)**((n+nn-m-mm)/2)*mp.sqrt((n+1)*(nn+1))*integration
    return float(result)

def theo_str(Phi,delta_r):
    def integrand(kappa):
        return kappa*Phi(kappa)*(1-mp.besselj(0,kappa*delta_r))
    integration = mp.quad(integrand,[0,mp.inf])
    result=integration*4*mp.pi
    return float(result)

if __name__ == '__main__':
    mp.mp.dps = 32
    f_plot=lambda x:theo_str(Kolmogorov(0.5),x)
    fx=[f_plot(x) for x in np.linspace(0,1,100)]
    # mp.plot(
    #     f_plot, [0, 1],
    # )
    plt.plot(np.linspace(0,1,100),fx)
    plt.show()
    # Von_Karman(1,1)
    # result= cov_j_j_prime(1,1,1,1,Kolmogorov(0.5),1)
    # print(result)