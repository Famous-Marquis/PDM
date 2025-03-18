"""
数据生成模块:

gen_data: 多线程并行产生模拟湍流数据

"""

import glob
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from aberration import ZERNIKE_NUMS, PhaseScreen
from beam import LG_mode

DATADIR = "./DDPM_data/"
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


def gen_data(args: tuple[int, int, list, int, int]):
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
    idx: int
        当前数据集的序号
    position: int
        用于显示,是进度条的显示位置
    """
    length: int = args[0]
    repeat_times = args[1]
    Dr0_range = args[2]
    if isinstance(Dr0_range, list):
        Dr0_list = np.random.uniform(Dr0_range[0], Dr0_range[1], (repeat_times,))
    else:
        Dr0_list = [Dr0_range for i in range(repeat_times)]
    idx = args[3]
    position = args[4]
    ps = PhaseScreen(256, ZERNIKE_NUMS, cache_dir="TURBULENCE/cache/")
    with tqdm(
            total=length * repeat_times,
            desc="当前生成第{}~{}组数据".format(
                idx * repeat_times, (idx + 1) * repeat_times
            ),
            position=position,
            dynamic_ncols=True,
    ) as pbar:
        list_of_samples = []
        for Dr0 in Dr0_list:
            for j in range(length):
                ps.simulate_turbulence(Dr0, method="zernike")
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
        assert len(list_of_samples) == length * repeat_times
        # 存储数据集
        pbar.close()
    series = pd.Series(list_of_samples)
    # plt.imshow(series[0])  # type: ignore
    # plt.colorbar()
    # plt.show()
    series.to_pickle(os.path.join(DATADIR, "data", "{}_DDPM.pkl".format(idx)))


def generate_data(length_per_Dr0, Dr0_range, parallel_processors=2,nums_Dr0=2):
    ##设置生成样本的参数
    assert (
            nums_Dr0 % parallel_processors == 0
    ), "样本集数量必须为PARALLEL_PROCESSORS的整数倍!"
    # 必须为 PARALLEL_PROCESSORS 的倍数

    if not os.path.exists(DATADIR):
        os.mkdir(DATADIR)
    if not os.path.exists(os.path.join(DATADIR, "data")):
        os.mkdir(os.path.join(DATADIR, "data"))
    mp.freeze_support()

    repeat_times = int(nums_Dr0 / parallel_processors)

    args_tuple = list(
        (
            (length_per_Dr0, repeat_times, Dr0_range, idx, idx)
            for idx in range(parallel_processors)
        )
    )
    with mp.Pool(parallel_processors) as pool:
        pool.map(gen_data, args_tuple)
    os.system("cls")
    # gen_data((100, 3, [5, 15], 2, 2))
    # 此处创建N个进程,同时执行生成数据函数. 在此阻塞，直到所有进程任务结束,执行下一行代码

    print("all have finished!!")

    """
    将所有生成的模拟湍流数据,进行整合,供gan模型使用
    """

    # 指定文件模式，找到所有以 '_gan.npy' 结尾的文件
    file_pattern = "*_DDPM.pkl"
    # 使用 glob.glob 查找匹配的文件
    file_list = glob.glob(os.path.join(DATADIR, "data", file_pattern))

    # 读取并合并文件
    # 所有文件都是相同形状的 NumPy 数组构成的Series
    merged_series = pd.Series(dtype=object)
    for file_name in file_list:
        series = pd.read_pickle(file_name)
        assert type(series) == pd.Series, "文件格式读取出来不是Series"
        merged_series = pd.concat([merged_series, series], ignore_index=True)

    assert len(merged_series) == len(series) * len(
        file_list
    ), "合成出来的数据元素数目不对"
    assert type(merged_series[0]) == np.ndarray, "Series中元素不是numpy数组"
    # assert list(merged_series[0].shape) == [256, 256], "Series中的元素大小不是256*256"
    assert list(merged_series[0].shape) == [
        ZERNIKE_NUMS,
    ]
    merged_series.to_pickle("./DDPM_data/merged_data.pkl")
    print(f'All "_DDPM.npy" files have been merged into "./DDPM_data/merged_data.pkl"')


if __name__ == "__main__":
    generate_data(length_per_Dr0=250, nums_Dr0=12,Dr0_range=1)
    # gen_data((25, 1, [5, 6], 1, 1))
