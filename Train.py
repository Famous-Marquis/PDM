import pandas as pd
from torchvision.utils import save_image
from pyexpat import model
from warmup_scheduler import GradualWarmupScheduler
from torchvision import transforms
import numpy as np
from typing import Dict
import torch
from torch.utils.data import Dataset, DataLoader
import os
from generate_data import generate_data
import warmup_scheduler
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
import torch.optim as optim
from tqdm import tqdm
from draw_record import save_plt_img


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
        return item  # .to(device=self.device)


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
        optimizer, T_max=model_config["epoch"], eta_min=0, last_epoch=-1
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
    for e in range(model_config["epoch"]):
        # 由于样本集太小，每次迭代后将重新生成样本集
        generate_data(
            length_per_Dr0=model_config["data_length_per_Dr0"],
            nums_Dr0=model_config["nums_Dr0"],
        )

        dataset = PandasDataset(
            "./DDPM_data/merged_data.pkl",
            "cuda:0",
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=model_config["batch_size"],
            shuffle=True,
            num_workers=3,
            drop_last=True,
            pin_memory=True,
        )

        with tqdm(dataloader, dynamic_ncols=True) as tqdm_dataloader:
            losses = []
            for images in tqdm_dataloader:
                optimizer.zero_grad()
                x_0 = images.double().to(device)
                loss = trainer(x_0).sum() / 100.0
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    net_model.parameters(), model_config["grad_clip"]
                )
                optimizer.step()
                tqdm_dataloader.set_postfix(
                    ordered_dict={
                        "epoch": e,
                        "loss": loss.item(),
                        "img shape": x_0.shape,
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
        }
        df = pd.DataFrame(record_dict)
        df.to_pickle("./Record/record.pkl")
        save_plt_img("./Record/record.pkl")


def eval(model_config: Dict):
    with torch.no_grad():
        device = torch.device(model_config["device"])
        model = UNet(
            T=model_config["T"],
            ch=model_config["channel"],
            ch_mult=model_config["channel_mult"],
            attn=model_config["attn"],
            nums_resblocks=model_config["nums_res_blocks"],
            dropout=model_config["dropout"],
        )
        ckpt = torch.load(
            os.path.join(
                model_config["save_weight_dir"], model_config["test_load_weight"]
            ),
            map_location=device,
        )
        model.load_state_dict(ckpt)
        print("model load weight done")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, model_config["beta_1"], model_config["beta_T"], model_config["T"]
        ).to(device)
        noisyImage = torch.randn(
            size=[model_config["batch_size"], 1, 256, 256],
        )
        saveNoise = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(
            saveNoise,
            os.path.join(
                model_config["sampled_dir"], model_config["sampledNoisyImgName"]
            ),
            nrow=model_config["nrow"],
        )
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5
        save_image(
            sampledImgs,
            os.path.join(model_config["sampled_dir"], model_config["sampledImgName"]),
            nrow=model_config["nrow"],
        )
