# Portions of this code are derived from unpublished work by Lu Chenda
# presented at OFC 2021 https://opg.optica.org/abstract.cfm?URI=OFC-2021-Th1A.16
# Copyright (c) 2021 The Author(s)
#
# These portions are used with permission.
# -------------------------------------------------------------------------------
# Copyright 2025 Beijing University of Posts and Telecommunications
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

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal
from tqdm import tqdm
from aberration import ZERNIKE_NUMS
from beam import LG_mode

DATADIR = "./Datasets/"
mode_list = [
    (1),
    (-2),
    (3),
    (-5),
    (1, -2),
    (1, 3),
    (1, -5),
    (-2, 3),
    (-2, 5),
    (3, -5),
    (1, -2, 3),
    (1, -2, -5),
    (1, 3, -5),
    (-2, 3, -5),
    (1, 3, -2, -5),
]


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


def OAM_beam(mode):
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
        return resize_beam(LG_mode(mode))

beam_list = [OAM_beam(mode) for mode in mode_list]

def gen_data(l0,L0,r0,R,alpha,ps,nums,name):
    """
    并行生成不同湍流强度(Dr0)的函数,参数按照元组形式传递

    Parameter
    -----------------
    length: int
        每组强度下,生成样本的数量
    repeat_times: int
        每次负责生成多少组
    dr0_range: list[low,high]
        湍流强度参数

    """

    with tqdm(
            total=nums,
            desc="当前生成数据",
            dynamic_ncols=True,
    ) as pbar:
        list_of_samples = []
        for _ in range(nums):
            ps.simulate_turbulence(l0,L0,r0,R,alpha, method="zernike")
            z_coes = np.array(ps.get_coeffients())
            assert list(z_coes.shape) == [
                ZERNIKE_NUMS,
            ], "z-coes形状不对！"

            # image = ps.get_screen()
            # plt.imshow(image)
            # plt.colorbar()
            # plt.show()
            # assert list(image.shape) == [256, 256], "生成的相位屏长宽不对!"
            # assert type(image) == np.ndarray
            # assert image.dtype == np.float64
            # list_of_samples.append(image)
            list_of_samples.append(z_coes)
            pbar.update(1)
        assert len(list_of_samples) == nums
        # 存储数据集
        pbar.close()
    arr = np.array(list_of_samples)
    # plt.imshow(series[0])  # type: ignore
    # plt.colorbar()
    # plt.show()
    np.save(name,arr)


def process_coeff_beam(coeff, beam, ps):
    ps.set_zernike_coeffients(coeff)
    psf = ps.get_psf()
    img = signal.fftconvolve(psf, beam)
    img = np.abs(img)
    img = resize_beam(img)
    return img


def gen_img_label(coeff_pickle_or_npy, name,ps,num):
    if coeff_pickle_or_npy[-4:] == ".pkl":
        data_series = pd.read_pickle(coeff_pickle_or_npy)
        data_matrix = np.stack(data_series, axis=0).astype(np.float32)
    elif coeff_pickle_or_npy[-4:] == ".npy":
        data_matrix = np.load(coeff_pickle_or_npy)
    labels = []
    imgs = []
    futures = []
    # bps = BatchPhaseScreen(1,256, ZERNIKE_NUMS, cache_dir="./cache/")
    with ThreadPoolExecutor() as executor:

        # psf_array = np.array([bps.get_psf(bps.set_zernike_coeffients(coeff)) for coeff in data_matrix])  # 预计算 PSF
        with tqdm(total=len(beam_list) * len(data_matrix[:num]),
                  desc=f'generating VGG batch_data of {name}...') as pbar:
            for i, beam in enumerate(beam_list):
                # 光束循环
                for j, coeff in enumerate(data_matrix[:num]):
                    # 系数循环
                    future = executor.submit(process_coeff_beam, coeff, beam, ps)
                    future.add_done_callback(lambda p: pbar.update(1))
                    futures.append((future, i))
            for future, i in futures:
                imgs.append(future.result())
                labels.append(i)

            img_stack = np.array(imgs, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.float32)
            np.savez_compressed(f"./Datasets/VGG_Datasets_{name}.npz", imgs=img_stack, labels=labels_array,
                     allow_pickle=False)
            pbar.close()



if __name__ == "__main__":

    ...