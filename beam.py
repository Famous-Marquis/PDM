"""
光束模块:

1. LG_mode
2. OAM_mode
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matlab import meshgrid, cart2pol
from PIL import Image


def LG_mode(
    l=4, N=1024, w=0.75e-2, wvl=0.532e-6, d1=2.65e-4, p=0, z=1000, need_coord=False
):
    """计算LG_mode"""
    k = 2 * np.pi / wvl

    x1, y1 = meshgrid(np.linspace(-N / 2, N / 2 - 1, N) * d1)
    phi, r = cart2pol(x1, y1)

    zR = np.pi * w * w / wvl
    wz = w * np.sqrt(1 + np.power(z / zR, 2))
    a = 2 * math.factorial(p) / (np.pi * math.factorial(p + abs(l)))

    LG1 = np.sqrt(a)
    LG2 = (1 / wz) * np.power(r * np.sqrt(2) / wz, abs(l))
    LG3 = np.exp(-r * r / (wz * wz))
    b = 2 * r * r / (wz * wz)

    if p == 0:
        LG4 = 1
    elif p == 1:
        LG4 = 2 - b
    elif p == 2:
        LG4 = (b * b - 6 * b + 6) / 2

    LG5 = np.exp(-1j * k * (r * r) * z / (2 * (np.power(z, 2) + np.power(zR, 2))))
    LG6 = np.exp(1j * (2 * p + abs(l) + 1) * np.arctan(z / zR))
    LG_abs = LG1 * LG2 * LG3 * LG4 * LG5 * LG6
    phase_OAM = phi * l
    LG = LG_abs * np.exp(-1j * phase_OAM)
    if need_coord:
        return LG, x1, y1
    else:
        return LG


def OAM_mode(mode, size=224):
    assert isinstance(mode, (tuple, int))

    def resize_beam(beam):
        """
        压缩beam至(224,224)
        """
        # 首先确保输入是 NumPy 数组
        if isinstance(beam, Image.Image):
            beam = np.array(beam)

        # 取绝对值
        beam_abs = np.abs(beam)

        # 将 NumPy 数组转换为 PIL Image 对象
        beam_image = Image.fromarray(np.uint8(beam_abs))

        # 使用 PIL 的 resize 方法来改变图像大小
        resized_image = beam_image.resize((224, 224))

        # 将 PIL Image 对象转换回 NumPy 数组
        resized_beam = np.array(resized_image)

        return resized_beam

    if type(mode) == tuple:
        if len(mode) == 2:
            l1, l2 = mode
            return resize_beam(LG_mode(l1) + LG_mode(l2))
        elif len(mode) == 3:
            l1, l2, l3 = mode
            return resize_beam(LG_mode(l1) + LG_mode(l2) + LG_mode(l3))
        elif len(mode) == 4:
            l1, l2, l3, l4 = mode
            return resize_beam(LG_mode(l1) + LG_mode(l2) + LG_mode(l3) + LG_mode(l4))
    else:

        return resize_beam(LG_mode(l=mode))  # type: ignore


if __name__ == "__main__":

    u1 = LG_mode()
    u2 = LG_mode(l=-3)
