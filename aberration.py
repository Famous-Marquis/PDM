"""
默认支持400维zernike系数
ZM和ZN是zernike系数的索引，其中ZM的索引有两种表示方式，根据需要取用
"""
CACHE_DIR="./cache"
ZERNIKE_NUMS = 64
SCREEN_SIZE = 224
ZN = [
    0,
    1,
    1,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    22,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
]

ZM_ = [
    0,
    1,
    -1,
    0,
    -2,
    2,
    -1,
    1,
    -3,
    3,
    0,
    2,
    -2,
    4,
    -4,
    1,
    -1,
    3,
    -3,
    5,
    -5,
    0,
    -2,
    2,
    -4,
    4,
    -6,
    6,
    -1,
    1,
    -3,
    3,
    -5,
    5,
    -7,
    7,
    0,
    2,
    -2,
    4,
    -4,
    6,
    -6,
    8,
    -8,
    1,
    -1,
    3,
    -3,
    5,
    -5,
    7,
    -7,
    9,
    -9,
    0,
    -2,
    2,
    -4,
    4,
    -6,
    6,
    -8,
    8,
    -10,
    10,
    -1,
    1,
    -3,
    3,
    -5,
    5,
    -7,
    7,
    -9,
    9,
    -11,
    11,
    0,
    2,
    -2,
    4,
    -4,
    6,
    -6,
    8,
    -8,
    10,
    -10,
    12,
    -12,
    1,
    -1,
    3,
    -3,
    5,
    -5,
    7,
    -7,
    9,
    -9,
    11,
    -11,
    13,
    -13,
    0,
    -2,
    2,
    -4,
    4,
    -6,
    6,
    -8,
    8,
    -10,
    10,
    -12,
    12,
    -14,
    14,
    -1,
    1,
    -3,
    3,
    -5,
    5,
    -7,
    7,
    -9,
    9,
    -11,
    11,
    -13,
    13,
    -15,
    15,
    0,
    2,
    -2,
    4,
    -4,
    6,
    -6,
    8,
    -8,
    10,
    -10,
    12,
    -12,
    14,
    -14,
    16,
    -16,
    1,
    -1,
    3,
    -3,
    5,
    -5,
    7,
    -7,
    9,
    -9,
    11,
    -11,
    13,
    -13,
    15,
    -15,
    17,
    -17,
    0,
    -2,
    2,
    -4,
    4,
    -6,
    6,
    -8,
    8,
    -10,
    10,
    -12,
    12,
    -14,
    14,
    -16,
    16,
    -18,
    18,
    -1,
    1,
    -3,
    3,
    -5,
    5,
    -7,
    7,
    -9,
    9,
    -11,
    11,
    -13,
    13,
    -15,
    15,
    -17,
    17,
    -19,
    19,
    0,
    2,
    -2,
    4,
    -4,
    6,
    -6,
    8,
    -8,
    10,
    -10,
    12,
    -12,
    14,
    -14,
    16,
    -16,
    18,
    -18,
    20,
    -20,
    1,
    -1,
    3,
    -3,
    5,
    -5,
    7,
    -7,
    9,
    -9,
    11,
    -11,
    13,
    -13,
    15,
    -15,
    17,
    -17,
    19,
    -19,
    21,
    -21,
    0,
    -2,
    2,
    -4,
    4,
    -6,
    6,
    -8,
    8,
    -10,
    10,
    -12,
    12,
    -14,
    14,
    -16,
    16,
    -18,
    18,
    -20,
    20,
    -22,
    22,
    -1,
    1,
    -3,
    3,
    -5,
    5,
    -7,
    7,
    -9,
    9,
    -11,
    11,
    -13,
    13,
    -15,
    15,
    -17,
    17,
    -19,
    19,
    -21,
    21,
    -23,
    23,
    0,
    2,
    -2,
    4,
    -4,
    6,
    -6,
    8,
    -8,
    10,
    -10,
    12,
    -12,
    14,
    -14,
    16,
    -16,
    18,
    -18,
    20,
    -20,
    22,
    -22,
    24,
    -24,
    1,
    -1,
    3,
    -3,
    5,
    -5,
    7,
    -7,
    9,
    -9,
    11,
    -11,
    13,
    -13,
    15,
    -15,
    17,
    -17,
    19,
    -19,
    21,
    -21,
    23,
    -23,
    25,
    -25,
    0,
    -2,
    2,
    -4,
    4,
    -6,
    6,
    -8,
    8,
    -10,
    10,
    -12,
    12,
    -14,
    14,
    -16,
    16,
    -18,
    18,
    -20,
    20,
    -22,
    22,
    -24,
    24,
    -26,
    26,
    -1,
    1,
    -3,
    3,
    -5,
    5,
    -7,
    7,
    -9,
    9,
    -11,
    11,
    -13,
    13,
    -15,
    15,
    -17,
    17,
    -19,
    19,
    -21,
    21,
]

ZM = [
    0,
    1,
    1,
    0,
    2,
    2,
    1,
    1,
    3,
    3,
    0,
    2,
    2,
    4,
    4,
    1,
    1,
    3,
    3,
    5,
    5,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    10,
    10,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    11,
    11,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    10,
    10,
    12,
    12,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    11,
    11,
    13,
    13,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    10,
    10,
    12,
    12,
    14,
    14,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    11,
    11,
    13,
    13,
    15,
    15,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    10,
    10,
    12,
    12,
    14,
    14,
    16,
    16,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    11,
    11,
    13,
    13,
    15,
    15,
    17,
    17,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    10,
    10,
    12,
    12,
    14,
    14,
    16,
    16,
    18,
    18,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    11,
    11,
    13,
    13,
    15,
    15,
    17,
    17,
    19,
    19,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    10,
    10,
    12,
    12,
    14,
    14,
    16,
    16,
    18,
    18,
    20,
    20,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    11,
    11,
    13,
    13,
    15,
    15,
    17,
    17,
    19,
    19,
    21,
    21,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    10,
    10,
    12,
    12,
    14,
    14,
    16,
    16,
    18,
    18,
    20,
    20,
    22,
    22,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    11,
    11,
    13,
    13,
    15,
    15,
    17,
    17,
    19,
    19,
    21,
    21,
    23,
    23,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    10,
    10,
    12,
    12,
    14,
    14,
    16,
    16,
    18,
    18,
    20,
    20,
    22,
    22,
    24,
    24,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    11,
    11,
    13,
    13,
    15,
    15,
    17,
    17,
    19,
    19,
    21,
    21,
    23,
    23,
    25,
    25,
    0,
    2,
    2,
    4,
    4,
    6,
    6,
    8,
    8,
    10,
    10,
    12,
    12,
    14,
    14,
    16,
    16,
    18,
    18,
    20,
    20,
    22,
    22,
    24,
    24,
    26,
    26,
    1,
    1,
    3,
    3,
    5,
    5,
    7,
    7,
    9,
    9,
    11,
    11,
    13,
    13,
    15,
    15,
    17,
    17,
    19,
    19,
    21,
    21,
]
from typing import Literal
import numpy as np
from matlab import meshgrid, cart2pol, ft2, ift2, randn
import matplotlib.pyplot as plt
from scipy import optimize
from math import factorial
import os


def GenerateZnAndZm(z_num=ZERNIKE_NUMS):
    N, M = [], []
    k, n, m = 0, 0, 0

    def append_nm(n, m, k):
        N.append(n)
        M.append(m)
        # print('index {}, n:{}, m:{}'.format(k, n, m))
        return k + 1

    for m in range(n + 1):
        if n >= m and abs(n - m) % 2 == 0:
            if m == 0:
                k = append_nm(n, m, k)
                if k >= z_num:
                    return N, M
            else:
                k = append_nm(n, m, k)
                if k >= z_num:
                    return N, M
                k = append_nm(n, m, k)
                if k >= z_num:
                    return N, M
        n += 1
    return ZN, ZM


## 点扩散函数
def Psf(Z):
    """
    Z: zernike序列(系数coes * 多项式polys)
    """
    abbe = np.exp(-1j * 2 * np.pi * Z)
    for i in range(len(abbe)):
        for j in range(len(abbe)):
            if abbe[i][j] == 1:
                abbe[i][j] = 0
    P = ft2(abbe, 1)
    P = P / P.max()
    return P


## 功率谱反演法生成相位屏
def FtPhaseScreen(Dr0, N, L0=float("inf"), l0=0):
    r0 = 1 / Dr0
    fx = np.arange(-N / 2, N / 2)

    fx, fy = meshgrid(fx)
    _, f = cart2pol(fx, fy)
    fm = 5.92 / l0 / (2 * np.pi) if l0 != 0 else float("inf")
    f0 = 1 / L0

    PSD_phi = (
            0.023 * r0 ** (-5 / 3) * np.exp(-((f / fm) ** 2)) / (f ** 2 + f0 ** 2) ** (11 / 6)
    )
    PSD_phi[int(N / 2), int(N / 2)] = 0

    cn = (randn(N) + 1j * randn(N)) * np.sqrt(PSD_phi)
    phz = ift2(cn, 1).real
    return phz


def FtShPhaseScreen(Dr0, N, L0=float("inf"), l0=0):
    r0 = 1 / Dr0
    D = 1
    phz_hi = FtPhaseScreen(Dr0, N, L0=float("inf"), l0=0)
    x, y = meshgrid(np.arange(-N / 2, N / 2) / N)
    phz_lo = np.zeros([N, N])

    for p in range(3):
        del_f = 1 / (3 ** p * D)
    # (忽视)TODO : 子谐波补充相位屏低频分量
    pass


## 计算第i阶的Zernike多项式
def ZernikePoly(i, r=None, theta=None):
    if np.all(r == None) or np.all(theta == None):
        x, y = meshgrid(np.linspace(-1, 1, 256))
        theta, r = cart2pol(x, y)
    n, m = ZN[i], ZM[i]
    pupil = r < 1  # type: ignore

    def _zrf(n, m, r):
        R = 0
        for s in range((n - m) // 2 + 1):
            num = (-1) ** s * factorial(n - s)
            denom = (
                    factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s)
            )
            R = R + num / denom * (r ** (n - 2 * s))

        return R

    if m == 0:
        Z = np.sqrt(n + 1) * _zrf(n, 0, r)
    else:
        if i % 2 == 0:
            Z = np.sqrt(2 * (n + 1)) * _zrf(n, m, r) * np.cos(m * theta)
        else:
            Z = np.sqrt(2 * (n + 1)) * _zrf(n, m, r) * np.sin(m * theta)
    return Z * pupil


## 计算Zernike系数
def noll_zernike_coeffients(z_num=ZERNIKE_NUMS, Dr0=7):
    import scipy.special as S
    from numpy import linalg as la

    n = ZN[:z_num]
    m = ZM[:z_num]
    nn = n
    mm = m
    C = np.zeros([len(n), len(m)])
    for i in range(1, len(n)):
        for j in range(1, len(m)):
            if m[i] == mm[j]:
                k = (
                        2.2698
                        * pow(-1, int((n[i] + nn[j] - 2 * n[i]) / 2))
                        * pow((n[i] + 1) * (nn[j] + 1), 0.5)
                )
                A = S.gamma(14 / 3)
                a = S.gamma((n[i] + nn[j] - 5 / 3) / 2)
                B = pow(2, 14 / 3)
                b = S.gamma((n[i] - nn[j] + 17 / 3) / 2)
                c = S.gamma((nn[j] - n[i] + 17 / 3) / 2)
                d = S.gamma((n[i] + nn[j] + 23 / 3) / 2)
                C[i, j] = pow(Dr0, 5 / 3) * k * a * A / b / c / d / B
            else:
                C[i, j] = 0
                continue
    u, s, v = la.svd(C[1:, 1:])
    rand = np.random.normal(size=z_num - 1)
    B = np.sqrt(s) * rand
    A = np.dot(u, B)

    zernike = []
    one = np.array([1])
    zernike[:] = one.astype("float64")
    zernike[1:] = A

    return zernike


class PhaseScreen:
    """
    Parameter
    ---------------
    N:
        pixels
    znum:
        Zernike Order
    """

    def __init__(self, N=SCREEN_SIZE, znum=ZERNIKE_NUMS, cache_dir=CACHE_DIR):
        self.N = N
        self.znum = znum
        self.x, self.y = meshgrid(np.linspace(-1, 1, N))
        self.theta, self.r = cart2pol(self.x, self.y)
        self._z_coes = [0 for _ in range(znum)]
        self._scr = np.zeros([N, N])
        try:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            cache_path = os.path.join(cache_dir, "zpoly_" + str(znum) + "_" + str(N) + ".npy")
            self._zpolys = np.load(cache_path)
            assert N == self._zpolys[0].shape[0] == self._zpolys[0].shape[1]
        except:
            print("load cache file fails, init zpolys all over:\n znum={},N={}".format(znum,N))
            self._zpolys = []
            for i in range(znum):
                self._zpolys.append(ZernikePoly(i, self.r, self.theta))
            try:
                np.save(cache_path, self._zpolys)
            except:
                print("save cache file fails")

    # getters

    def get_zpoly(self, i):
        """
        获取第i阶的zernike多项式

        Parameter
        --------------
        i: int
            想要获取的第i阶的泽尼克系数
        """
        assert i > 0 and i <= self.znum
        return self._zpolys[i - 1]

    def get_zpolys(self):
        """获取全部的zernike多项式"""
        return self._zpolys

    def get_coeffients(self):
        """获取zernike系数"""
        return self._z_coes

    def get_screen(self):
        """获取相位屏"""
        return self._scr

    def get_psf(self):
        """获取点扩散函数"""
        return Psf(self.get_screen())

    # setters

    def set_screen(self, scr):
        """设置给定的相位屏"""
        assert scr.shape == (self.N, self.N)
        self._scr = scr
        return self

    def set_zernike_coeffients(self, z_coes, update_scr=True):
        """
        设置zernike系数.

        Parameter
        -------------------
        z_coes: List
            zernike系数列表

        """
        if isinstance(z_coes, list) or isinstance(z_coes, np.ndarray):
            if len(z_coes) < self.znum:
                z_coes += [0 for _ in range(self.znum - len(z_coes))]
            assert len(z_coes) == self.znum
            self._z_coes = z_coes
        else:
            raise NotImplementedError

        if update_scr:
            self.update_screen()
        return self

    def update_screen(self):
        """更新相位屏."""
        scr = np.zeros([self.N, self.N])
        assert (
                len(self._z_coes) == len(self._zpolys) == self.znum
        ), "len(_z_coes)={},len(_zploys)={},znum={}".format(
            len(self._z_coes), len(self._zpolys), self.znum
        )
        for i in range(self.znum):
            scr += self._z_coes[i] * self._zpolys[i]
        self._scr = scr
        return self

    def simulate_turbulence(self, Dr0, method: Literal["ft", "zernike"] = "zernike"):
        """
        进行大气湍流仿真,生成相位屏.

        Parameter
        ---------------
        Dr0: float
            大气湍流参数
        method: str, Literal['ft','zernike']
            大气湍流的生成方法: `'ft'`: 功率谱反演法(冯·卡曼谱)  `'zernike'`: zernike多项式

        """
        if method == "ft":
            scr = FtPhaseScreen(Dr0, self.N, L0=float("inf"), l0=0)
            self.fit(scr)
            self.update_screen()
        elif method == "zernike":
            z = noll_zernike_coeffients(z_num=self.znum, Dr0=Dr0)
            self.set_zernike_coeffients(z, update_scr=True)
        else:
            raise NotImplementedError

        return self

    def remove_tip_tilt(self):
        """
         去除屏幕显示中的倾斜效应（Tip-Tilt Effect）。

        Tip-Tilt Effect 通常指的是光学成像中的倾斜或偏移效应，这可能是由于设备的指向误差造成的。
        此方法通过将二次项系数设置为0来纠正这种效应，并更新屏幕显示。

         :return: 返回类的实例自身，允许链式调用。
        """
        # 将二次项系数设置为0，以去除倾斜
        self._z_coes[1] = 0  # 假设这是x的系数
        self._z_coes[2] = 0  # 假设这是y的系数

        # 更新屏幕显示以反映系数的变化
        self.update_screen()

        # 返回类的实例自身
        return self

    def show(self, dpi=60):
        """
        显示屏幕内容，通常是一个二维数组的可视化。

        parameter
        ---------
        dpi: 图像的每英寸点数，影响图像的清晰度。默认为60。
        """
        # 获取屏幕内容
        scr = self._scr

        # 创建一个新的图形，并设置其大小和分辨率
        plt.figure(figsize=(12, 8), dpi=dpi)

        # 显示屏幕内容，使用“bone”颜色映射来增强对比度
        plt.imshow(scr, cmap="bone")

        # 添加一个颜色条以便于理解图像的数值范围
        plt.colorbar(shrink=0.83)

        # 隐藏x轴和y轴的刻度
        plt.xticks([])
        plt.yticks([])

    def show_3d(self, dpi=60):
        """
        展示一个三维图形，其中Z轴代表屏幕的值。

        :param dpi: 图像的每英寸点数，影响图像的清晰度。默认为60。
        """
        # 创建一个新的图形，并设置其大小和分辨率
        fig = plt.figure(figsize=(12, 8), dpi=dpi)

        # 获取图形的轴，并设置为3D投影
        ax = fig.gca(projection="3d")

        # 生成X和Y轴的值，用于创建网格
        X = np.linspace(-1, 1, 224)
        Y = np.linspace(-1, 1, 224)
        X, Y = np.meshgrid(X, Y)

        # 假设self._scr是一个二维数组，用作Z轴的值
        Z = self._scr

        # 绘制3D曲面图
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap="RdYlGn",  # 颜色映射
            linewidth=0,  # 线条宽度
            antialiased=False,  # 是否抗锯齿
            alpha=0.6,  # 透明度
        )

        # 添加颜色条以便于理解图像的数值范围
        fig.colorbar(surf, shrink=0.6, aspect=30)

        # 显示图形
        plt.show()

    def fit(self, Z, znum=ZERNIKE_NUMS):
        """
        拟合屏幕数据到一个多项式，以去除倾斜效应。

        :param Z: 屏幕数据的二维数组。
        :param znum: 拟合多项式的阶数，默认为400。
        :return: 返回类的实例自身，允许链式调用。
        """
        # 定义一个遮罩，表示光学系统的孔径
        pupil = self.r < 1

        # 应用遮罩到屏幕数据
        Z = Z * pupil

        # 初始化拟合系数列表
        fit_coes = []

        # 确保输入数据的维度与类属性N匹配
        assert Z.shape[0] == self.N

        # 对每个多项式项进行拟合
        for i in range(znum):
            ZF = self._zpolys[i]  # 假设self._zpolys是多项式的系数或形状
            # 计算拟合系数，这里使用了简化的数学公式
            coe = sum(sum(Z * ZF)) * 2 * 2 / self.N / self.N / np.pi  # type: ignore
            fit_coes.append(round(coe, 3))  # 保留三位小数

        # 更新类属性_z_coes，包括拟合系数和未拟合的零系数
        self._z_coes = fit_coes + [0 for _ in range(self.znum - znum)]

        # 返回类的实例自身
        return self

    def add_lens(self, k, f):
        """添加滤镜

        Parameter
        -------------
        k,f : 滤镜参数
        """
        # lens = np.exp(-1j*k/(2*f)*(self.r**2))
        lens = k / (2 * f) * (self.r ** 2)
        self._scr = self.get_screen() + lens
        return self

    def add_pupil(self):
        """添加瞳孔(模拟镜头成像)"""
        pupil = self.r < 1
        assert self._scr is not None
        self._scr *= pupil
        return self


if __name__ == "__main__":
    N = 256
    ph = FtPhaseScreen(1, N)
    z = PhaseScreen(N=64)
    z.fit(ph)
