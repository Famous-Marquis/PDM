# Portions of this code are derived from unpublished work by Lu Chenda
# presented at OFC 2021 https://opg.optica.org/abstract.cfm?URI=OFC-2021-Th1A.16
# Copyright (c) 2021 The Author(s)
#
# These portions are used with permission.
#----------------------------------------------------------
# Copyright (c) 2025 Beijing University of Posts and Telecommunications
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from scipy.io import savemat
from tqdm import tqdm
from PSD import cov_j_j_prime, modified_Von_Karman
CACHE_DIR = "./cache"
N_DEGREE = 10
ZERNIKE_NUMS = int((N_DEGREE+1)*(N_DEGREE+2)/2)
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
import os
from math import factorial, gamma
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from matlab import meshgrid, cart2pol, ft2, ift2, randn


def GenerateZnAndZm(z_num=ZERNIKE_NUMS):
    N, M = [], []
    k, n, m = 0, 0, 0

    def append_nm(n, m, k):
        N.append(n)
        M.append(m)
        # print('index {}, n:{}, size:{}'.format(k, n, size))
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
        x, y = meshgrid(np.linspace(-1, 1, SCREEN_SIZE))
        theta, r = cart2pol(x, y)
    n, m = ZN[i], ZM_[i]
    pupil = r < 1  # type: ignore

    def _zrf(n, m, r):
        R = 0
        for s in range((n - abs(m)) // 2 + 1):
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
    m = ZM_[:z_num]

    C = np.zeros([z_num, z_num])
    for i in range(1, z_num):
        for j in range(1, z_num):
            if m[i] == m[j] and (not (i % 2 == j % 2) or m[i] == 0):
                k = (
                        2.2698
                        * (-1) ** ((n[i] + n[j] - 2 * m[i]) / 2)
                        * np.sqrt((n[i] + 1) * (n[j] + 1))
                )
                a = S.gamma((n[i] + n[j] - 5 / 3) / 2)
                b = S.gamma((n[i] - n[j] + 17 / 3) / 2)
                c = S.gamma((n[j] - n[i] + 17 / 3) / 2)
                d = S.gamma((n[i] + n[j] + 23 / 3) / 2)
                C[i, j] = Dr0 ** (5 / 3) * k * a / (b * c * d)

    # SVD-based sampling
    u, s, v = la.svd(C[2:, 2:])
    rand = np.random.normal(size=z_num - 1)
    print(u.shape, s.shape, rand.shape)
    A = np.dot(u, np.sqrt(s) * rand)

    # Insert piston = 1.0
    zernike = np.zeros(z_num)
    zernike[0] = 1.0
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
        self.zernike_coeff_generator = ZernikeCoefficientGenerator()
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
            print("load cache file fails, init zpolys all over:\n znum={},N={}".format(znum, N))
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

        Parameters
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

    def simulate_turbulence(self, l0,L0,r0,R,alpha, method: Literal["ft", "zernike"] = "zernike", ):
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
            raise NotImplementedError
        elif method == "zernike":
            z_coes_stack = self.zernike_coeff_generator.generate(self.znum,l0,L0,r0,R,alpha)
            self.set_zernike_coeffients(z_coes_stack)
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
        k,model_struct : 滤镜参数
        """
        # lens = np.exp(-1j*k/(2*model_struct)*(self.r**2))
        lens = k / (2 * f) * (self.r ** 2)
        self._scr = self.get_screen() + lens
        return self

    def add_pupil(self):
        """添加瞳孔(模拟镜头成像)"""
        pupil = self.r < 1
        assert self._scr is not None
        self._scr *= pupil
        return self

class ZernikeCoefficientGenerator:
    def __init__(self):
        self.Jn = None
        self.Jm = None
        self.params_cached = None
        self.C_cached = None
        self.u_cached = None
        self.s_cached = None

    def prepare(self, z_num, l0, L0, r0, R,alpha):
        """准备协方差矩阵C和SVD"""
        if (self.params_cached == [l0, L0, r0, R]) and (self.C_cached is not None):
            # 如果Dr0没变，直接跳过
            return
        print('initialize the coviriance matrix...')
        from numpy import linalg as la
        self.Jn, self.Jm = cal_n_m(n_degree=N_DEGREE)
        n = self.Jn
        m = abs(self.Jm)
        # print(len(n), len(m))
        # 计算新的C
        C = np.zeros([z_num + 1, z_num + 1])

        # print(z_num, C.shape)
        with tqdm(total=z_num*z_num,position=0, leave=True) as pbar:
            for i in range(1, z_num + 1):
                for j in range(1, z_num + 1):
                    pbar.update(1)
                    if m[i] == m[j] and (abs(j - i) % 2 == 0 or m[i] == 0):
                        # k = (
                        #         2.2698
                        #         * (-1) ** ((n[i] + n[j] - 2 * m[i]) / 2)
                        #         * np.sqrt((n[i] + 1) * (n[j] + 1))
                        # )
                        # a = S.gamma((n[i] + n[j] - 5 / 3) / 2)
                        # b = S.gamma((n[i] - n[j] + 17 / 3) / 2)
                        # c = S.gamma((n[j] - n[i] + 17 / 3) / 2)
                        # d = S.gamma((n[i] + n[j] + 23 / 3) / 2)
                        # C[i, j] = Dr0 ** (5 / 3) * k * a / (b * c * d)
                        C[i, j] = cov_j_j_prime(n[i], n[j], m[i], m[j], modified_Von_Karman(l0, L0, r0,alpha),
                                                R)
        C[1, 1] = 0

        diag = np.diag(C)

        # 计算 sqrt(C[i,i])，形状是 (n,)
        sqrt_diag = np.sqrt(diag)

        # 外积形成 sqrt(C[i,i] * C[j,j]) 矩阵，形状是 (n, n)
        denom = np.outer(sqrt_diag, sqrt_diag)
        denom[denom == 0] = 1
        # 归一化协方差矩阵
        C_normalized = C / denom
        # print(model_struct"C11,{C[1, 1]}")
        # print(model_struct"C22,{C[2, 2]}")

        plt.imshow(C_normalized[2:35, 2:35])
        # plt.yscale("log")
        plt.colorbar()
        plt.show()
        # SVD分解
        plt.plot(C[2:, 2:].diagonal())
        plt.yscale("log")
        # plt.xscale("log")
        plt.ylabel(r"$||a_j||^2$")
        plt.xlabel("Zernike index")
        plt.title("Non-Kolmogorov")
        plt.show()
        u, s, v = la.svd(C[2:, 2:])
        savemat("C.mat",{"Cov":C[1:,1:],"Cov_normalized":C_normalized[1:,1:],"C_diag":C[1:,1:].diagonal()})

        # 保存
        self.params_cached = [l0, L0, r0, R]
        self.C_cached = C
        self.u_cached = u
        self.s_cached = s

    def generate(self, z_num, l0, L0, r0, R,alpha):
        """生成批量Zernike系数"""
        self.prepare(z_num, l0, L0, r0, R,alpha)  # 先确保准备好了

        # 批量生成
        rand = np.random.normal(size=(z_num - 1,))  # [ znum-1,]
        A = np.dot(rand * np.sqrt(self.s_cached), self.u_cached.T)  # [znum-1,]

        zernike_coeff = np.zeros((z_num,))
        zernike_coeff[0] = 0.
        zernike_coeff[1:] = A

        return zernike_coeff


class BatchPhaseScreen(PhaseScreen):
    def __init__(self, batch, N=SCREEN_SIZE, znum=ZERNIKE_NUMS, cache_dir=CACHE_DIR):
        # todo：灵活batch，针对剩余的几个也能正常处理
        super().__init__(N, znum, cache_dir)
        self.batch = batch
        self.zernike_coeff_generator = ZernikeCoefficientGenerator()
        self.z_coes_stack = np.zeros([self.batch, self.znum])
        self.scr_stack = np.zeros([self.batch, self.N, self.N])
        try:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            cache_path = os.path.join(cache_dir,
                                      "zpoly_" + str(self.znum) + "_" + str(self.N) + ".npy")
            self._zpolys = np.load(cache_path)
            assert len(self._zpolys.shape) == 3
            assert self.N == self._zpolys.shape[1] == self._zpolys.shape[2]
        except FileNotFoundError:
            print("load cache file fails, init zpolys all over:\n znum={},N={}".format(znum, N))
            self._zpolys = []
            for i in range(znum):
                self._zpolys.append(ZernikePoly(i, self.r, self.theta))
            self._zpolys = np.array(self._zpolys)
            try:
                assert len(self._zpolys.shape) == 3
                assert self.N == self._zpolys.shape[1] == self._zpolys.shape[2]
                np.save(cache_path, self._zpolys)
            except Exception as e:
                print("save cache file fails:", e)

    def get_psf(self):
        return Psf(self.get_screen()[0])

        # raise NotImplementedError()

    def get_screen(self):
        return self.scr_stack

    def set_screen(self, screen_stack):
        if screen_stack.shape == (self.batch, self.N, self.N):
            self.scr_stack = screen_stack
        else:
            raise ValueError
        return self

    def set_zernike_coeffients(self, z_coes, update_scr=True):
        if isinstance(z_coes, np.ndarray):
            assert len(z_coes.shape) == 2, z_coes.shape
            batch = z_coes.shape[0]
            if z_coes.shape[1] < self.znum:
                padding = np.zeros([batch, self.znum - z_coes.shape[1]])
                z_coes = np.concatenate([z_coes, padding], axis=1)
            assert z_coes.shape[1] == self.znum
            self.z_coes_stack = z_coes
            self.batch = self.z_coes_stack.shape[0]
        else:
            raise NotImplementedError
        if update_scr:
            self.update_screen()

    def update_screen(self):
        """
        批量计算相位屏

        self.z_coes_stack: [batch,znum]
        self._zpoly: [N,N]
        """
        assert self._zpolys.shape[0] == self.znum, self._zpolys.shape
        z_coes_expand = self.z_coes_stack[:, :, None, None]
        self.scr_stack = np.sum(z_coes_expand * self._zpolys[None, :, :, :], axis=1)
        assert self.scr_stack.shape == (self.batch, self.N, self.N), self.scr_stack.shape
        return self

    def simulate_turbulence(self, Dr0, method: Literal["ft", "zernike"] = "zernike"):
        if method == "ft":
            raise NotImplementedError
        elif method == "zernike":
            z_coes_stack = self.zernike_coeff_generator.generate(self.batch, self.znum, Dr0)
            self.set_zernike_coeffients(z_coes_stack)
        else:
            raise NotImplementedError
        return self


def C_real(alpha=11 / 3):
    result = (
            1 / (2 * math.pi) ** 2
            * gamma(alpha - 1)
            * math.cos(alpha * math.pi / 2)
    )
    return result


def C_phi(alpha):
    result = (
            2 ** alpha
            * (gamma((alpha + 4) / 2)) ** 2
            * gamma((alpha + 6) / 2)
            * gamma((alpha + 2) / 2)
            * math.sin(math.pi * alpha / 2)
            / math.pi ** (alpha + 2)
            / gamma(alpha + 3)
    )

    return result


def A_beta(beta=11 / 3):
    result = (
            2 ** (beta - 2)
            * (gamma((beta + 2) / 2)) ** 2
            * gamma((beta + 4) / 2)
            * gamma(beta / 2)
            * math.sin(math.pi * (beta - 2) / 2)
            / math.pi ** beta
            / gamma(beta + 1)
    )
    return result


def cal_n_m(n_degree):
    n_degree += 1
    maxj = int(n_degree * (n_degree + 1) / 2)
    J_n = np.zeros((maxj + 1,), dtype=int)
    J_m = np.zeros((maxj + 1,), dtype=int)
    for n in range(n_degree):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:

                j = (
                        n * (n + 1) / 2
                        + abs(m)
                )
                if m > 0 and (n % 4 == 0 or n % 4 == 1):
                    ...
                if m < 0 and (n % 4 == 2 or n % 4 == 3):
                    ...
                if m >= 0 and (n % 4 == 2 or n % 4 == 3):
                    j += 1
                if m <= 0 and (n % 4 == 0 or n % 4 == 1):
                    j += 1
                j = int(j)
                J_n[j] = n
                J_m[j] = m
    return J_n, J_m


if __name__ == "__main__":
    # N = 256
    # ph = FtPhaseScreen(1, N)
    # z = PhaseScreen(N=64)
    # z.fit(ph)
    # ps=PhaseScreen()
    # ps.simulate_turbulence(Dr0=15)
    # phi=ps.get_screen()
    # plt.imshow(phi)
    # plt.show()
    # print(phi)
    # bps = BatchPhaseScreen(batch=1, N=SCREEN_SIZE)
    # bps.simulate_turbulence(1)
    ps=PhaseScreen()
    ps.simulate_turbulence(0.005,10,0.1,0.5,14/3)
    print(C_phi(5 / 3))
    print(C_real(11 / 3))
    print(A_beta(11 / 3))
    # print(A_alpha(11 / 3))
    Jn, Jm = cal_n_m(5)
    print(abs(Jm))
    print(Jn)
    # print(B(5/3))
    # print(0.49/(2*math.pi)**(5/3))
