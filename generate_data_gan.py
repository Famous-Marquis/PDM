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

import csv
import multiprocessing as mp
from typing import Union
import numpy as np
from scipy import signal
from PIL import Image
from tqdm import tqdm
from aberration import PhaseScreen
from beam import LG_mode

DATADIR = "../batch_data/gan/"
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
    压缩beam至(256,256)
    """
    # 首先确保输入是 NumPy 数组
    if isinstance(beam, Image.Image):
        beam = np.array(beam)

    # 取绝对值
    beam_abs = np.abs(beam)

    # 将 NumPy 数组转换为 PIL Image 对象
    beam_image = Image.fromarray(np.uint8(beam_abs))

    # 使用 PIL 的 resize 方法来改变图像大小
    resized_image = beam_image.resize((256, 256))

    # 将 PIL Image 对象转换回 NumPy 数组
    resized_beam = np.array(resized_image)

    return resized_beam


def gen_beam(mode):
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


beam_list = [gen_beam(mode) for mode in mode_list]


def gen_data(args: tuple[str, int, str, Union[int, float], int]):
    prefix: str = args[0]
    length: int = args[1]
    DATA_DIR: str = args[2]
    dr0 = args[3]
    position = args[4]

    def gen_data_per_Dr0(ps: PhaseScreen, Dr0, position):
        """
        按照强度生成数据的函数
        --------------------
        Dr0: 大气湍流参数(Dr0)
        """

        csv_name = prefix + "_" + str(position) + ".csv"
        csv_path = os.path.join(DATA_DIR, csv_name)
        with tqdm(total=length * len(beam_list), position=position) as pbar:
            # 创建一个文件夹，存储当前强度的数据
            coeffs_stack = np.zeros((length * len(beam_list), 64), dtype=np.float64)
            with open(csv_path, "w") as csv_file:
                writer = csv.writer(csv_file)
                coeffs_idx = 0
                # 为每一个光束进行【length】次大气湍流仿真
                for j, beam in enumerate(beam_list):
                    for i in range(length):
                        ps.simulate_turbulence(Dr0)
                        psf = ps.get_psf()
                        image = signal.fftconvolve(psf, beam, "same")
                        z_coes = ps.get_coeffients()
                        coeffs_stack[coeffs_idx] = np.array(z_coes)
                        coeffs_idx += 1
                        image_path = os.path.join(DATA_DIR, csv_name[:-4])
                        if not os.path.exists(image_path):
                            os.makedirs(image_path)
                        Image.fromarray(abs(image)).convert("RGB").save(
                            DATA_DIR
                            + "/"
                            + csv_name[:-4]
                            + "/"
                            + "{}.jpg".format(j * length + i)
                        )
                        writer.writerow(
                            [
                                DATA_DIR
                                + "/"
                                + csv_name[:-4]
                                + "/"
                                + "{}.jpg".format(j * length + i),
                                z_coes,
                                j,
                            ]
                        )
                        pbar.update(1)
            np.save("../batch_data/gan/{}_gan.npy".format(str(position)), coeffs_stack)

    ps = PhaseScreen(256, 64, "../cache")
    # print("generating...", prefix)
    gen_data_per_Dr0(ps=ps, Dr0=dr0, position=position)


import os


def merge_npy_files(directory=DATADIR, output_filename="gan.npy", pattern="_gan.npy"):
    # 获取指定目录下所有符合模式的.npy文件
    npy_files = sorted([f for f in os.listdir(directory) if f.endswith(pattern)])

    if not npy_files:
        print("No matching .npy files found.")
        return

    # 读取并合并所有.npy文件
    arrays = [np.load(os.path.join(directory, f)) for f in npy_files]
    merged_array = np.concatenate(arrays, axis=0)
    print(merged_array.shape)
    # 保存合并后的数组
    np.save(os.path.join(directory, output_filename), merged_array)
    print(f"Merged {len(npy_files)} files into {output_filename} in directory {directory}.")

if __name__ == "__main__":
    mp.freeze_support()
    # args_tuple = list(
    #     (("train", 5, DATADIR, 5 + dr0 / 10, i) for i, dr0 in enumerate(range(10)))
    # )
    args_tuple = list(
        (("train", 5, DATADIR, 1, i) for i in range(4))
    )
    with mp.Pool(2) as pool:
        pool.map(gen_data, args_tuple)
    # 此处阻塞，直到所有进程任务结束
    merge_npy_files()
    print("all have finished!!")

