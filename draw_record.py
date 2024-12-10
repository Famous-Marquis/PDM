"""
用于随时查看当前训练过程中,每次迭代的记录
"""

import matplotlib.axes
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import sys


# 读取对应数据
def save_plt_img(record_df_path):
    df = pd.read_pickle(record_df_path)
    df: pd.DataFrame
    record_dict = df.to_dict()
    loss_mean_per_epoch = [item for item in record_dict["loss_mean_per_epoch"].values()]
    loss_var_per_epoch = [
        item / 100 for item in record_dict["loss_var_per_epoch"].values()
    ]
    learning_rate_per_epoch = [
        item for item in record_dict["learning_rate_per_epoch"].values()
    ]

    epoch = [epoch for epoch in range(len(loss_mean_per_epoch))]

    # 绘图
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epoch, loss_mean_per_epoch, "b-o", label="loss_mean/100")
    ax1.plot(epoch, loss_var_per_epoch, "r--^", label="loss_var/100")
    ax1.set_title("Loss - epoch")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epoch, learning_rate_per_epoch, "g-o", label="learning_rate")
    ax2.set_title("Learning_rate - epoch")
    ax2.set_xlabel("epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("./Record/record.jpg")


save_plt_img("./Record/record-11.24-night.pkl")
