import torch
import numpy as np
from einops import rearrange


def mse_loss(img, gt):
    img = img.view(gt.shape)
    return ((img - gt) ** 2).mean()


class EffDistLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, m, interval):
        '''
        Efficient O(N) realization of distortion loss.
        There are B rays each with N sampled points.
        w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
        m:        Float tensor in shape [B,N]. Midpoint distance to camera of each point.
        interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
        '''
        n_rays = np.prod(w.shape[:-1])
        wm = (w * m)
        w_cumsum = w.cumsum(dim=-1)
        wm_cumsum = wm.cumsum(dim=-1)

        w_total = w_cumsum[..., [-1]]
        wm_total = wm_cumsum[..., [-1]]
        w_prefix = torch.cat([torch.zeros_like(w_total), w_cumsum[..., :-1]], dim=-1)
        wm_prefix = torch.cat([torch.zeros_like(wm_total), wm_cumsum[..., :-1]], dim=-1)
        loss_uni = (1/3) * interval * w.pow(2)
        loss_bi = 2 * w * (m * w_prefix - wm_prefix)
        if torch.is_tensor(interval):
            ctx.save_for_backward(w, m, wm, w_prefix, w_total, wm_prefix, wm_total, interval)
            ctx.interval = None
        else:
            ctx.save_for_backward(w, m, wm, w_prefix, w_total, wm_prefix, wm_total)
            ctx.interval = interval
        ctx.n_rays = n_rays
        return (loss_bi.sum() + loss_uni.sum()) / n_rays

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        interval = ctx.interval
        n_rays = ctx.n_rays
        if interval is None:
            w, m, wm, w_prefix, w_total, wm_prefix, wm_total, interval = ctx.saved_tensors
        else:
            w, m, wm, w_prefix, w_total, wm_prefix, wm_total = ctx.saved_tensors
        grad_uni = (1/3) * interval * 2 * w
        w_suffix = w_total - (w_prefix + w)
        wm_suffix = wm_total - (wm_prefix + wm)
        grad_bi = 2 * (m * (w_prefix - w_suffix) + (wm_suffix - wm_prefix))
        grad = grad_back * (grad_bi + grad_uni) / n_rays
        return grad, None, None, None


eff_distloss = EffDistLoss.apply


def distortion_loss(weights, z_vals, near, far):
    # loss from mip-nerf 360; efficient implementation from DVGOv2 (https://github.com/sunset1995/torch_efficient_distloss) with some modifications

    # weights: [B, N, n_samples, 1]
    # z_vals: [B, N, n_samples, 1]

    assert weights.shape == z_vals.shape
    assert len(weights.shape) == 4
    weights = rearrange(weights, "b n s 1 -> (b n) s")
    z_vals = rearrange(z_vals, "b n s 1 -> (b n) s")

    # go from z space to s space (for linear sampling; INVERSE SAMPLING NOT IMPLEMENTED)
    s = (z_vals - near) / (far - near)

    # distance between samples
    interval = s[:, 1:] - s[:, :-1]

    loss = eff_distloss(weights[:, :-1], s[:, :-1], interval)
    return loss


def occupancy_loss(weights):
    # loss from lolnerf (prior on weights to be distributed as a mixture of Laplacian distributions around mode 0 or 1)
    # weights: [B, N, n_samples, 1]
    assert len(weights.shape) == 4

    pw = torch.exp(-torch.abs(weights)) + torch.exp(
        -torch.abs(torch.ones_like(weights) - weights)
    )
    loss = -1.0 * torch.log(pw).mean()
    return loss
