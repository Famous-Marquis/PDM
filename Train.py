import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
from aberration import ZERNIKE_NUMS, PhaseScreen, CACHE_DIR
from draw_record import save_plt_img_config
from generate_data import generate_data
from model_config import MODEL_CONFIG


class NumpyDataset(Dataset):
    def __init__(self, datapath, transform=None):
        super().__init__()
        self.transform = transform
        self.data = np.load(datapath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx])
        if self.transform:
            item = self.transform(torch.from_numpy(sample))
        return item


class PandasDataset(Dataset):

    def __init__(self, datapath: str, device, transform=None):
        super().__init__()
        self.series = pd.read_pickle(datapath)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.series)

    # 取出数据,并进行转换的操作在__getitem__中实现
    def __getitem__(self, idx):
        sample = torch.from_numpy(self.series[idx])[None, :]
        assert type(sample) == torch.Tensor
        if self.transform != None:
            item = self.transform(sample)
        else:
            item = sample
        return item  # .to(device=self.device)


def batch_cos_sim(A, B):
    B=np.squeeze(B)
    assert A.shape == B.shape
    dot_product = np.sum(A * B, axis=1)
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    assert norm_A.shape==(100,),"norm_A:".format(norm_A.shape)
    cos_sim = dot_product / (norm_A * norm_B)
    return cos_sim[:5]


# TODO:在训练期间，增加指标评估，用于监控训练


def train(model_config: Dict):
    device = torch.device(model_config["device"])

    # 准备模型
    net_model = (
        UNet(
            T=model_config["T"],
            ch=model_config["channel"],
            ch_mult=model_config["channel_mult"],
            attn=model_config["attn"],
            nums_resblocks=model_config["num_res_blocks"],
            dropout=model_config["dropout"],
        )
        .to(device)
        .double()
    )
    if model_config["training_load_weight"]:
        R = net_model.load_state_dict(
            torch.load(
                os.path.join(
                    model_config["save_weight_dir"],
                    model_config["training_load_weight"],
                ),
                map_location=device,
            )
        )
        print(R)
    # 优化器
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=model_config["learning_rate"], weight_decay=1e-4
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=model_config["epoch"],
        eta_min=model_config["learning_rate"] / 2,
        last_epoch=-1,
    )
    warm_up_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=model_config["multiplier"],
        total_epoch=model_config["epoch"] // 10,
        after_scheduler=cosine_scheduler,
    )
    trainer = GaussianDiffusionTrainer(
        net_model, model_config["beta_1"], model_config["beta_T"], model_config["T"]
    ).to(device)
    # 初始化这些用于记录的变量
    learning_rate_per_epoch = []
    loss_mean_per_epoch = []
    loss_var_per_epoch = []
    # 开始训练
    generate_data(
        Dr0_range=model_config["Dr0_range"],
        length_per_Dr0=model_config["data_length_per_Dr0"],
        nums_Dr0=model_config["nums_Dr0"],
    )

    dataset = PandasDataset(
        "./DDPM_data/merged_data.pkl",
        "cuda:0",
        # transform=transforms.Compose(
        #     [
        #         # transforms.RandomHorizontalFlip(),
        #         # transforms.ToTensor(),
        #         # transforms.Normalize((0.5,), (0.5,)),
        #     ]
        # ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    for e in range(model_config["epoch"]):
        # //由于样本集太小，每次迭代后将重新生成样本集
        with tqdm(dataloader, dynamic_ncols=True) as tqdm_dataloader:
            losses = []
            for coeff in tqdm_dataloader:
                optimizer.zero_grad()
                x_0 = coeff.double().to(device)
                loss = trainer(x_0).sum()
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    net_model.parameters(), model_config["grad_clip"]
                )
                optimizer.step()
                tqdm_dataloader.set_postfix(
                    ordered_dict={
                        "epoch": e,
                        "loss": loss.item(),
                        "sample shape": x_0.shape,
                        "LR": optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )
                losses.append(loss.item())
        # 记录每次训练的loss及学习率
        loss_mean_per_epoch.append(np.mean([losses]))
        loss_var_per_epoch.append(np.var([losses]))
        learning_rate_per_epoch.append(optimizer.state_dict()["param_groups"][0]["lr"])
        assert len(loss_mean_per_epoch) == e + 1
        warm_up_scheduler.step()
        # 保存本次epoch的参数
        torch.save(
            net_model.state_dict(),
            os.path.join(model_config["save_weight_dir"], "ckpt_" + str(e) + "_.pt"),
        )
        # 将每次训练记录的loss的均值、方差、学习率保存下来
        record_dict = {
            "loss_mean_per_epoch": loss_mean_per_epoch,
            "loss_var_per_epoch": loss_var_per_epoch,
            "learning_rate_per_epoch": learning_rate_per_epoch,
            # "model_config": model_config,
        }
        df = pd.DataFrame(record_dict)
        df.to_pickle("./Record/28.pkl")
        save_plt_img_config("Record/28.pkl")


def eval(model_config: Dict):
    with (torch.no_grad()):
        device = torch.device(model_config["device"])
        model = UNet(
            T=model_config["T"],
            ch=model_config["channel"],
            ch_mult=model_config["channel_mult"],
            attn=model_config["attn"],
            nums_resblocks=model_config["num_res_blocks"],
            dropout=model_config["dropout"],
        )
        ckpt = torch.load(
            os.path.join(
                model_config["save_weight_dir"], model_config["test_load_weight"]
            ),
            map_location=device,
        )
        model.load_state_dict(ckpt)
        print("model load weight done, \n load_state:{}".format(model_config["test_load_weight"]))
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, model_config["beta_1"], model_config["beta_T"], model_config["T"]
        ).to(device)
        ##
        sample_nums = 100
        noisy_coeffs = torch.randn(
            size=[sample_nums, 1, ZERNIKE_NUMS],
        ).to(device)
        # saveNoise = torch.clamp(noisy_coeffs * 0.5 + 0.5, 0, 1)
        # save_image(
        #     saveNoise,
        #     os.path.join(
        #         model_config["sampled_dir"], model_config["sampledNoisyImgName"]
        #     ),
        #     nrow=model_config["nrow"],
        # )
        sampled_coeffs = sampler(noisy_coeffs)
        # sampledImgs = sampledImgs * 0.5 + 0.5
        sampled_coeffs = sampled_coeffs.cpu()
        sampled_coeffs = sampled_coeffs.numpy()
        assert sampled_coeffs.shape[2] == ZERNIKE_NUMS
        ps = PhaseScreen(256, ZERNIKE_NUMS, cache_dir=CACHE_DIR)
        sampled_coes = np.zeros([sampled_coeffs.shape[0], sampled_coeffs.shape[2]])
        fig, axs = plt.subplots(2, 3, figsize=(16, 6))
        for i, z_coes in enumerate(sampled_coeffs):
            sampled_coes[i] = z_coes[0]

            if i < 6:
                ax = axs.flat[i]
                # ps.set_zernike_coeffients(list(z_coes[0]))
                im = ax.imshow(z_coes, aspect="auto", cmap=plt.get_cmap("viridis"))
                ax.set_yticks([])
                plt.colorbar(im, ax=ax)
        fig.suptitle("sampled_data")
        plt.tight_layout()
        plt.savefig("./SampledImgs/sample.jpg")
        plt.close(fig)
        fig, axs = plt.subplots(2, 3, figsize=(16, 6))
        for ax in axs.flat:
            ps.simulate_turbulence(MODEL_CONFIG["Dr0_range"])
            coeff = ps.get_coeffients()
            coeff = np.array(coeff)
            coeff = coeff[None, :]
            im = ax.imshow(coeff, aspect="auto", cmap=plt.get_cmap("viridis"))
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
        fig.suptitle("real_data")
        plt.tight_layout()
        plt.savefig("./SampledImgs/real.jpg")
        plt.close(fig)

        real_coes = np.zeros([sampled_coeffs.shape[0], sampled_coeffs.shape[2]])
        for i in range(sample_nums):
            ps.simulate_turbulence(MODEL_CONFIG["Dr0_range"])
            coeff = ps.get_coeffients()
            coeff = np.array(coeff)[None, :]
            real_coes[i] = coeff[0]

        assert real_coes.shape == np.squeeze(sampled_coeffs).shape, "形状不同,real:{},sampled:{}".format(
            real_coes.shape, np.squeeze(sampled_coeffs).shape)
        cos_sim = batch_cos_sim(real_coes, sampled_coeffs)
        print("生成数据与原始样本的余弦相似度 \n[10个为例]：")
        for i,item in enumerate(cos_sim[:10]):
            print(i,":", item)
        return sampled_coes
        # save_image(
        #     sampledImgs,
        #     os.path.join(model_config["sampled_dir"], model_config["sampledImgName"]),
        #     nrow=model_config["nrow"],
        # )
