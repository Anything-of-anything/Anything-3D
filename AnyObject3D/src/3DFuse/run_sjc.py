import math
import numpy as np
import torch
import torch.nn as nn
import cv2
from einops import rearrange
from imageio import imwrite
from pydantic import validator

from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
    get_event_storage, get_heartbeat, read_stats
)
from my.config import BaseConf, dispatch, optional_load_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter, karras_t_schedule
from run_img_sampling import GDDPM, SD, StableDiffusion
from misc import torch_samps_to_imgs
from pose import PoseConfig

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis


device_glb = torch.device("cuda")


def tsr_stats(tsr):
    return {
        "mean": tsr.mean().item(),
        "std": tsr.std().item(),
        "max": tsr.max().item(),
    }


class SJC(BaseConf):
    family:     str = "sd"
    gddpm:      GDDPM = GDDPM()
    sd:         SD = SD(
        variant="v1",
        prompt="A high quality photo of a delicious burger",
        scale=100.0
    )
    lr:         float = 0.05
    n_steps:    int = 10000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=True, bg_texture_hw=4,
        bbox_len=1.0
    )
    pose:       PoseConfig = PoseConfig(rend_hw=64, FoV=60.0, R=1.5)

    emptiness_scale:    int = 10
    emptiness_weight:   int = 1e4
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    depth_weight:       int = 0

    var_red:     bool = True

    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()

        family = cfgs.pop("family")
        model = getattr(self, family).make()

        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        sjc_3d(**cfgs, poser=poser, model=model, vox=vox)


def sjc_3d(
    poser, vox, model: ScoreAdapter,
    lr, n_steps, emptiness_scale, emptiness_weight, emptiness_step, emptiness_multiplier,
    depth_weight, var_red, **kwargs
):
    del kwargs

    assert model.samps_centered()
    _, target_H, target_W = model.data_shape()
    bs = 1
    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)
    opt = torch.optim.Adamax(vox.opt_params(), lr=lr)

    H, W = poser.H, poser.W
    Ks, poses, prompt_prefixes = poser.sample_train(n_steps)

    ts = model.us[30:-10]
    fuse = EarlyLoopBreak(5)

    # same_noise = torch.randn(1, 4, H, W, device=model.device).repeat(bs, 1, 1, 1)
    n_steps=200
    with tqdm(total=n_steps) as pbar, \
        HeartBeat(pbar) as hbeat, \
            EventStorage() as metric:
        for i in range(n_steps):
            if fuse.on_break():
                break

            p = f"{prompt_prefixes[i]} {model.prompt}"
            score_conds = model.prompts_emb([p])
            # text_z = model.get_text_embeds([p],[""])

            score_conds['c']=score_conds['c'].repeat(bs,1,1)
            score_conds['uc']=score_conds['uc'].repeat(bs,1,1)

            y, depth, ws = render_one_view(vox, aabb, H, W, Ks[i], poses[i], return_w=True)

            if isinstance(model, StableDiffusion):
                pass
            else:
                y = torch.nn.functional.interpolate(y, (target_H, target_W), mode='bilinear')

            opt.zero_grad()

            with torch.no_grad():
                chosen_σs = np.random.choice(ts, bs, replace=False)
                chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
                chosen_σs = torch.as_tensor(chosen_σs, device=model.device, dtype=torch.float32)
                # chosen_σs = us[i]

                noise = torch.randn(bs, *y.shape[1:], device=model.device)

                zs = y + chosen_σs * noise

                Ds = model.denoise(zs, chosen_σs, **score_conds)

                if var_red:
                    grad = (Ds - y) / chosen_σs
                else:
                    grad = (Ds - zs) / chosen_σs

                grad = grad.mean(0, keepdim=True)
                
            y.backward(-grad, retain_graph=True)

            if depth_weight > 0:
                center_depth = depth[7:-7, 7:-7]
                border_depth_mean = (depth.sum() - center_depth.sum()) / (64*64-50*50)
                center_depth_mean = center_depth.mean()
                depth_diff = center_depth_mean - border_depth_mean
                depth_loss = - torch.log(depth_diff + 1e-12)
                depth_loss = depth_weight * depth_loss
                depth_loss.backward(retain_graph=True)

            emptiness_loss = torch.log(1 + emptiness_scale * ws).mean()
            emptiness_loss = emptiness_weight * emptiness_loss
            if emptiness_step * n_steps <= i:
                emptiness_loss *= emptiness_multiplier
            emptiness_loss.backward()

            opt.step()

            metric.put_scalars(**tsr_stats(y))

            if every(pbar, percent=1):
                with torch.no_grad():
                    if isinstance(model, StableDiffusion):
                        y = model.decode(y)
                    # print(y.shape)
                    # print(depth.shape)
                    vis_routine(metric, y, depth)

            # if every(pbar, step=2500):
            #     metric.put_artifact(
            #         "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
            #     )
            #     with EventStorage("test"):
            #         evaluate(model, vox, poser)

            metric.step()
            pbar.update()
            pbar.set_description(p)
            hbeat.beat()

        metric.put_artifact(
            "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
        )
        with EventStorage("test"):
            evaluate(model, vox, poser)

        metric.step()

        hbeat.done()


@torch.no_grad()
def evaluate(score_model, vox, poser):
    H, W = poser.H, poser.W
    vox.eval()
    K, poses = poser.sample_test(100)

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()
    hbeat = get_heartbeat()

    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)

    num_imgs = len(poses)

    for i in (pbar := tqdm(range(num_imgs))):
        if fuse.on_break():
            break

        pose = poses[i]
        y, depth = render_one_view(vox, aabb, H, W, K, pose)
        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        vis_routine(metric, y, depth)

        metric.step()
        hbeat.beat()
    metric.step()


def render_one_view(vox, aabb, H, W, K, pose, return_w=False):
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)
    # print(ro.shape)
    ro, rd, t_min, t_max = scene_box_filter(ro, rd, aabb)
    
    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)
    rgbs, depth, weights = render_ray_bundle(vox, ro, rd, t_min, t_max)

    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H, w=W)
    if return_w:
        return rgbs, depth, weights
    else:
        return rgbs, depth


def scene_box_filter(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth):
    pane = nerf_vis(y, depth, final_H=256)
    im = torch_samps_to_imgs(y)[0]
    
    # depth_ = torch.nn.functional.interpolate(
    #     depth.unsqueeze(dim=0).unsqueeze(dim=0), (512,512), mode='bilinear', antialias=True
    #     )
   
    depth_pt = depth.squeeze().clone()
    mask=(depth_pt<5)
    # import pdb; pdb.set_trace()

    depth_pt = -1* depth_pt
    depth_pt -= torch.min(depth_pt)
    depth_pt /= torch.max(depth_pt)


    depth_pt = depth_pt.cpu().numpy()
    bg_th=0.01
    depth_np = -1*depth.squeeze()
    depth_np[mask] -= torch.min(depth_np[mask])
    depth_np[mask] /= torch.max(depth_np[mask])
    depth_np[~mask] = torch.min(depth_np[mask])
    depth_np=depth_np.cpu().numpy()
    # depth_np = np.log(1. + depth_np + 1e-12)
    x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, scale=1000, ksize=3)
    y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, scale=1000,ksize=3)
    z = np.ones_like(x) * 2*np.pi
    x[depth_pt < bg_th] = 0
    y[depth_pt < bg_th] = 0
    normal = np.stack([x, y, z], axis=2)
    normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
    normal=np.array(torch.nn.functional.interpolate(torch.from_numpy(normal).permute(2,0,1).unsqueeze(dim=0),(512,512),mode='bilinear').squeeze().cpu().permute(1,2,0))
    normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)


    depth = depth.cpu().numpy()
    metric.put_artifact("normal",'.png',"",lambda fn: imwrite(fn, normal_image))
    metric.put_artifact("view", ".png", "",lambda fn: imwrite(fn, pane))
    metric.put_artifact("img", ".png", "",lambda fn: imwrite(fn, im))
    metric.put_artifact("depth", ".npy","", lambda fn: np.save(fn, depth))


def evaluate_ckpt():
    cfg = optional_load_config(fname="full_config.yml")
    assert len(cfg) > 0, "can't find cfg file"
    mod = SJC(**cfg)

    family = cfg.pop("family")
    model: ScoreAdapter = getattr(mod, family).make()
    vox = mod.vox.make()
    poser = mod.pose.make()

    pbar = tqdm(range(1))

    with EventStorage(), HeartBeat(pbar):
        ckpt_fname = latest_ckpt()
        state = torch.load(ckpt_fname, map_location="cpu")
        vox.load_state_dict(state)
        vox.to(device_glb)

        with EventStorage("test"):
            evaluate(model, vox, poser)


def latest_ckpt():
    ts, ys = read_stats("./", "ckpt")
    assert len(ys) > 0
    return ys[-1]


if __name__ == "__main__":
    seed_everything(0)
    dispatch(SJC)
    # evaluate_ckpt()
