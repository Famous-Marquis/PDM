import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from model_config import MODEL_CONFIG

GROUPDIVIDE = MODEL_CONFIG["batch_size"] / 2


# B,C,H,W
class Swish(nn.Module):
    def forward(self, X):
        return X + torch.sigmoid(X)


class SeqAttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.q_proj = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.k_proj = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.v_proj = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        ############################
        if MODEL_CONFIG["norm"] == "GroupNorm":
            self.norm = nn.GroupNorm(GROUPDIVIDE, in_ch)
        elif MODEL_CONFIG["norm"] == "BatchNorm":
            self.norm = nn.BatchNorm1d(in_ch)
        elif MODEL_CONFIG["norm"] is None:
            self.norm = None
        ############################
        self.proj = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)  # type:ignore
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, L = x.shape
        if self.norm is not None:
            h = self.norm(x)
        else:
            h = x
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.permute(0, 2, 1)
        assert list(q.shape) == [B, L, C], "q.shape incorrect!!"
        q = q.view(B, L, C)

        k = k.view(B, C, L)

        w = torch.bmm(q, k) * (int(C) ** (-0.5))

        assert list(w.shape) == [B, L, L], "w.shape incorrect!!"
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 1)
        assert list(v.shape) == [B, L, C]
        v = v.view(B, L, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, L, C]
        h = h.view(B, L, C).permute(0, 2, 1)

        h = self.proj(h)
        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential()
        if MODEL_CONFIG["norm"] == "GroupNorm":
            self.block1.add_module("GroupNorm", nn.GroupNorm(GROUPDIVIDE, in_ch))
        elif MODEL_CONFIG["norm"] == "BatchNorm":
            self.block1.add_module("BatchNorm", nn.BatchNorm1d(in_ch))
        elif MODEL_CONFIG["norm"] is None:
            ...
        self.block1.add_module("Swish", Swish())
        self.block1.add_module("Conv1d", nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1), )

        self.temb_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
        # ?time_embedding 的输出？？
        self.block2 = nn.Sequential()
        if MODEL_CONFIG["norm"] == "GroupNorm":
            self.block2.add_module("GroupNorm", nn.GroupNorm(GROUPDIVIDE, out_ch))
        elif MODEL_CONFIG["norm"] == "BatchNorm":
            self.block2.add_module("BatchNorm", nn.BatchNorm1d(out_ch))
        elif MODEL_CONFIG["norm"] is None:
            ...
        self.block2.add_module("Dropout", nn.Dropout(dropout))
        self.block2.add_module("Swish", Swish())
        self.block2.add_module("Conv1d", nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1), )
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1, padding=0)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn_block = SeqAttnBlock(out_ch)
        else:
            self.attn_block = nn.Identity()

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)  # type:ignore
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)  # type: ignore

    def forward(self, X, temb):
        Y = self.block1(X)
        Y += self.temb_proj(temb)[:, :, None]
        Y = self.block2(Y)

        Y += self.shortcut(X)
        Y = self.attn_block(Y)

        return Y


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, tdim):
        assert d_model % 2 == 0
        super().__init__()
        omega = torch.arange(0, d_model, 2) / d_model * math.log(10000)
        omega = torch.exp(-omega)

        pos = torch.arange(T).float()
        omega = pos[:, None] * omega[None, :]

        assert list(omega.shape) == [T, d_model / 2]
        omega = torch.stack([torch.sin(omega), torch.cos(omega)], dim=-1)
        assert list(omega.shape) == [T, d_model / 2, 2]

        omega = omega.view(T, d_model)
        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(omega),
            nn.Linear(d_model, tdim),
            Swish(),
            nn.Linear(tdim, tdim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        if __name__ == "__main__":
            t = t.int()
        emb = self.time_embedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)  # type:ignore

    def forward(self, X, temb):
        X = self.main(X)
        return X


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)  # type:ignore

    def forward(self, X, temb):
        X = F.interpolate(X, scale_factor=2, mode="nearest")
        X = self.main(X)
        return X


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, nums_resblocks, dropout) -> None:
        super().__init__()
        assert all([i <= len(ch_mult) for i in attn]), "attn index out of bound"
        tdim = ch * 4

        self.time_embedding = TimeEmbedding(T, ch, tdim)
        ###########~ 可能需要一层卷积初始化,使得通道数变为偶数!
        self.head = nn.Conv1d(1, ch, 3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult

            for _ in range(nums_resblocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=now_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = out_ch
                chs.append(now_ch)

            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList(
            [
                ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
                ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
            ]
        )

        self.upblocks = nn.ModuleList()

        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(nums_resblocks + 1):
                self.upblocks.append(
                    ResBlock(
                        in_ch=now_ch + chs.pop(),
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = out_ch

            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        assert len(chs) == 0
        self.tail = nn.Sequential()
        if MODEL_CONFIG["norm"] == "GroupNorm":
            self.tail.add_module("GroupNorm", nn.GroupNorm(GROUPDIVIDE, now_ch))
        elif MODEL_CONFIG["norm"] == "BatchNorm":
            self.tail.add_module("BatchNorm", nn.BatchNorm1d(now_ch))
        elif MODEL_CONFIG["norm"] is None:
            ...
        self.tail.add_module("Swish", Swish())
        self.tail.add_module("Conv1d",nn.Conv1d(now_ch, 1, 3, stride=1, padding=1))
        self.initialize()
        # summary(self,(300,1,ZERNIKE_NUMS))

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)  # type:ignore
        init.xavier_uniform_(self.tail[-1].weight)  # type:ignore
        init.zeros_(self.tail[-1].bias)  # type:ignore

    def forward(self, X, t):
        temb = self.time_embedding(t)

        h = self.head(X)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        for layer in self.middleblocks:
            h = layer(h, temb)
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


class WrappedUNet(nn.Module):
    def __init__(self, model, T):
        super(WrappedUNet, self).__init__()
        self.model = model
        self.T = T  # 传入时间步长范围

    def forward(self, x):
        x = x.double()
        batch_size = x.shape[0]
        t = torch.randint(self.T, (batch_size,), device=x.device, dtype=torch.int)  # 生成随机 t
        return self.model(x, t).double()  # 传入原始 UNet


if __name__ == "__main__":
    ...

# ?点扩散函数，可以从像素点提取吗？
# *可以
