from Train import train, eval

MODEL_CONFIG = {
    # 运行模式&加载CheckPoint设置：
    "state": "eval",  # "train" or "eval"
    # 是否加载某次的权重?
    "training_load_weight": None,  # None or "ckpt_X_.pt"
    # 用于生成测试用例权重是:
    "test_load_weight": 'ckpt_49_.pt',
    # 模型的超参数
    "beta_1": 1e-4,  # β1~βT是一系列线性的参数. 是训练时叠加噪声的强度
    "beta_T": 0.005,
    "T": 640,  # 噪声去除一共需要经历T个步骤
    "channel": 2,  # 预测噪声时,模型从原始图像(1层通道)提取到的初始通道数  **必须为偶数**
    "channel_mult": [
        1,
        2,
        3
    ],  # 其长度决定了UNet的层数; 第X层UNet每次采样后的通道数 = channel_mult*channel.
    "attn": [0, 1, 2],  # UNet中,哪几个层使用注意力机制,索引从0开始
    "num_res_blocks": 1,  # UNet每层,由几个残差块构成
    "dropout": 0.15,  # 训练时的Dropout比率
    # 训练相关参数
    "epoch": 50,  # 总的迭代次数
    "batch_size": 30,  # 每批图片的个数
    "learning_rate": 5e-4,  # 预设学习率. 学习率由WarmUp调度器&Cosine调度器组成.
    "multiplier": 2.0,  # 初始调度器经历几次迭代后, 学习率逐渐调整为multiplier*learning_rate. 随后由余弦调度器进行一次涨落
    "grad_clip": 1.0,  # 参数的最大梯度(防止过拟合/难收敛)
    # 每次迭代后，生成新数据的参数
    "Dr0_range": 1,  # CONST or List
    "nums_Dr0": 10,  # [2的倍数] 每次更新数据库,生成几种Dr0的数据
    "data_length_per_Dr0": 30,  # 每个Dr0生成多少个图片; 总图片量 = nums_Dr0 * data_length_per_Dr0
    # 常规配置（不用动）
    "save_weight_dir": "./Checkpoints/",
    "sampled_dir": "./SampledImgs/",
    "device": "cuda:0",
    "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    "sampledImgName": "SampledNoGuidenceImgs.png",
    "nrow": 8,
}


def main(model_config=None):
    if model_config is None:
        model_config = MODEL_CONFIG
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
        return None
    else:
        return eval(modelConfig)


if __name__ == "__main__":
    z_coes_array =main(MODEL_CONFIG)
    if z_coes_array is not None:
        print(z_coes_array.shape)
    else:
        print(None)
