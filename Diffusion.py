import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm


def extract(v: torch.Tensor, t: torch.Tensor, x_shape):
    """
    提取特定时间步的系数,并重塑为适合广播的形状

    """
    device = t.device
    out = torch.gather(v, 0, t).float().to(device)
    return out.view(
        [t.shape[0]] + [1] * (len(x_shape) - 1)
    )  # 将后面的维度设置为1,用于广播


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T).double())
        # TODO:尝试余弦调度的β值
        self.beta: torch.Tensor
        alpha = 1 - self.beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.sqrt_alpha_bar: torch.Tensor
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar))
        self.sqrt_one_minus_alpha_bar: torch.Tensor

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        eps = torch.randn_like(x_0)
        x_t = (
                      extract(self.sqrt_alpha_bar, t, x_0.shape) * x_0
                      + extract(self.sqrt_one_minus_alpha_bar, t, x_0.shape)
              ) * eps
        loss = (
                functional.mse_loss(self.model(x_t, t), eps, reduction="none")
                / x_0.shape[0]
        )
        return loss


# 数值溢出，采用对数化简
class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.T = T
        self.model = model
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T).double())
        self.beta: torch.Tensor

        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = functional.pad(self.alpha_bar, (1, 0), value=1)[:-1]

        self.register_buffer("coeff1", 1 / torch.sqrt(self.alpha))
        self.coeff1: torch.Tensor
        self.register_buffer(
            "coeff2",
            -1
            / torch.sqrt(self.alpha)
            * (1 - self.alpha)
            / torch.sqrt(1 - self.alpha_bar),
        )
        self.coeff2: torch.Tensor

        self.register_buffer(
            "posterior_var",
            self.beta * (1 - self.alpha_bar_prev) / (1 - self.alpha_bar),
        )
        self.posterior_var: torch.Tensor

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (extract(self.coeff1, t, x_t.shape) * x_t) - extract(
            self.coeff2, t, x_t.shape
        ) * eps

    def p_mean_variance(self, x_t, t):
        var = torch.cat([self.posterior_var[1:2], self.beta[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        x_t = x_T
        for time_step in tqdm(reversed(range(self.T)), total=self.T, desc="sampling"):
            # print(time_step)
            t = (
                    x_t.new_ones(
                        [
                            x_t.shape[0],
                        ],
                        dtype=torch.long,
                    )
                    * time_step
            )
            mean, var = self.p_mean_variance(x_t, t)

            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor"

        x_0 = x_t
        return torch.clip(x_0, -1, 1)
