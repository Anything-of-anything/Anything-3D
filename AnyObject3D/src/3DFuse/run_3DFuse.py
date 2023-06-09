import os
import numpy as np
import torch
from einops import rearrange
from imageio import imwrite
from pydantic import validator
import imageio
import gradio as gr

from PIL import Image

from my.utils import (tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
                      get_event_storage, get_heartbeat, read_stats)
from my.config import BaseConf, dispatch, optional_load_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter
from run_img_sampling import SD
from misc import torch_samps_to_imgs
from pose import PoseConfig

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (as_torch_tsrs, rays_from_img, ray_box_intersect,
                            render_ray_bundle)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis

from pytorch3d.renderer import PointsRasterizationSettings

from semantic_coding import semantic_coding, semantic_karlo, semantic_sd
from pc_project import point_e, render_depth_from_cloud

device_glb = torch.device("cuda")


def tsr_stats(tsr):
    return {
        "mean": tsr.mean().item(),
        "std": tsr.std().item(),
        "max": tsr.max().item(),
    }


class SJC_3DFuse(BaseConf):
    family: str = "sd"
    sd: SD = SD(variant="v1",
                prompt="a comfortable bed",
                scale=100.0,
                dir="./results",
                alpha=0.3)
    lr: float = 0.05
    n_steps: int = 10000
    vox: VoxConfig = VoxConfig(model_type="V_SD",
                               grid_size=100,
                               density_shift=-1.0,
                               c=3,
                               blend_bg_texture=False,
                               bg_texture_hw=4,
                               bbox_len=1.0)
    pose: PoseConfig = PoseConfig(rend_hw=64, FoV=60.0, R=1.5)

    emptiness_scale: int = 10
    emptiness_weight: int = 1e4
    emptiness_step: float = 0.5
    emptiness_multiplier: float = 20.0

    depth_weight: int = 0

    var_red: bool = True
    exp_dir: str = "./results"
    ti_step: int = 800
    pt_step: int = 800
    initial: str = ""
    random_seed: int = 0
    semantic_model: str = "Karlo"
    bg_preprocess: bool = True
    num_initial_image: int = 4

    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()
        seed = cfgs.pop('random_seed')
        seed_everything(seed)
        initial = cfgs.pop('initial')
        exp_instance_dir = os.path.join(cfgs.pop('exp_dir'), initial)

        initial_prompt = cfgs['sd']['prompt']
        semantic_model = cfgs.pop('semantic_model')

        # Initial image generation
        image_dir = os.path.join(exp_instance_dir, 'initial_image')

        if semantic_model == "Karlo":
            semantic_karlo(initial_prompt, image_dir,
                           cfgs['num_initial_image'], cfgs['bg_preprocess'],
                           seed)
        elif semantic_model == "SD":
            semantic_sd(initial_prompt, image_dir, cfgs['num_initial_image'],
                        cfgs['bg_preprocess'], seed)
        else:
            raise NotImplementedError

        # Optimization  and pivotal tuning for LoRA
        semantic_coding(exp_instance_dir, cfgs, self.sd, initial)

        # Load SD with Consistency Injection Module
        family = cfgs.pop("family")
        model = getattr(self, family).make()
        print(model.prompt)
        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        # Get coarse point cloud from off-the-shelf model
        points = point_e(device=device_glb, exp_dir=exp_instance_dir)

        # Score distillation
        next(
            fuse_3d(**cfgs,
                    poser=poser,
                    model=model,
                    vox=vox,
                    exp_instance_dir=exp_instance_dir,
                    points=points,
                    is_gradio=False))

    def run_gradio(self, points, exp_instance_dir):
        cfgs = self.dict()
        initial = cfgs.pop('initial')
        # exp_dir=os.path.join(cfgs.pop('exp_dir'),initial)

        # Optimization  and pivotal tuning for LoRA
        yield gr.update(
            value=None
        ), "Tuning for the LoRA layer is starting now. It will take approximately ~10 mins.", gr.update(
            value=None)
        semantic_coding(exp_instance_dir, cfgs, self.sd, initial)

        # Load SD with Consistency Injection Module
        family = cfgs.pop("family")
        model = getattr(self, family).make()
        print(model.prompt)
        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        # Score distillation
        yield from fuse_3d(**cfgs,
                           poser=poser,
                           model=model,
                           vox=vox,
                           exp_instance_dir=exp_instance_dir,
                           points=points,
                           is_gradio=True)

    def run_offline(self, points, exp_instance_dir):
        cfgs = self.dict()
        initial = cfgs.pop('initial')
        # exp_dir=os.path.join(cfgs.pop('exp_dir'),initial)

        # Optimization  and pivotal tuning for LoRA
        # yield gr.update(value=None), "Tuning for the LoRA layer is starting now. It will take approximately ~10 mins.", gr.update(value=None)
        semantic_coding(exp_instance_dir, cfgs, self.sd, initial)

        # Load SD with Consistency Injection Module
        family = cfgs.pop("family")
        model = getattr(self, family).make()
        print(model.prompt)
        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()
        # Score distillation
        next(
            fuse_3d(**cfgs,
                    poser=poser,
                    model=model,
                    vox=vox,
                    exp_instance_dir=exp_instance_dir,
                    points=points,
                    is_gradio=False))


def fuse_3d(poser, vox, model: ScoreAdapter, lr, n_steps, emptiness_scale,
            emptiness_weight, emptiness_step, emptiness_multiplier,
            depth_weight, var_red, exp_instance_dir, points, is_gradio,
            **kwargs):
    del kwargs
    if is_gradio:
        yield gr.update(
            visible=True
        ), "LoRA layers tuning has just finished. \nScore distillation has started.", gr.update(
            visible=True)

    assert model.samps_centered()
    _, target_H, target_W = model.data_shape()
    bs = 1
    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)
    opt = torch.optim.Adamax(vox.opt_params(), lr=lr)

    H, W = poser.H, poser.W
    Ks_, poses_, prompt_prefixes_, angles_list = poser.sample_train(
        n_steps, device_glb)

    ts = model.us[30:-10]

    fuse = EarlyLoopBreak(5)

    raster_settings = PointsRasterizationSettings(image_size=800,
                                                  radius=0.02,
                                                  points_per_pixel=10)

    ts = model.us[30:-10]
    calibration_value = 0.0



    with tqdm(total=n_steps) as pbar, \
        HeartBeat(pbar) as hbeat, \
            EventStorage(output_dir=os.path.join(exp_instance_dir,'3d')) as metric:

        for i in range(len(poses_)):
            if fuse.on_break():
                break

            depth_map = render_depth_from_cloud(points, angles_list[i],
                                                raster_settings, device_glb,
                                                calibration_value)

            y, depth, ws = render_one_view(vox,
                                           aabb,
                                           H,
                                           W,
                                           Ks_[i],
                                           poses_[i],
                                           return_w=True)

            p = f"{prompt_prefixes_[i]} {model.prompt}"
            score_conds = model.prompts_emb([p])

            score_conds['c'] = score_conds['c'].repeat(bs, 1, 1)
            score_conds['uc'] = score_conds['uc'].repeat(bs, 1, 1)

            opt.zero_grad()

            with torch.no_grad():
                chosen_σs = np.random.choice(ts, bs, replace=False)
                chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
                chosen_σs = torch.as_tensor(chosen_σs,
                                            device=model.device,
                                            dtype=torch.float32)

                noise = torch.randn(bs, *y.shape[1:], device=model.device)

                zs = y + chosen_σs * noise

                Ds = model.denoise(zs, chosen_σs, depth_map.unsqueeze(dim=0),
                                   **score_conds)

                if var_red:
                    grad = (Ds - y) / chosen_σs
                else:
                    grad = (Ds - zs) / chosen_σs

                grad = grad.mean(0, keepdim=True)

            y.backward(-grad, retain_graph=True)

            if depth_weight > 0:
                center_depth = depth[7:-7, 7:-7]
                border_depth_mean = (depth.sum() -
                                     center_depth.sum()) / (64 * 64 - 50 * 50)
                center_depth_mean = center_depth.mean()
                depth_diff = center_depth_mean - border_depth_mean
                depth_loss = -torch.log(depth_diff + 1e-12)
                depth_loss = depth_weight * depth_loss
                depth_loss.backward(retain_graph=True)

            emptiness_loss = torch.log(1 + emptiness_scale * ws).mean()
            emptiness_loss = emptiness_weight * emptiness_loss
            if emptiness_step * n_steps <= i:
                emptiness_loss *= emptiness_multiplier
            emptiness_loss.backward()

            opt.step()

            metric.put_scalars(**tsr_stats(y))

            if every(pbar, percent=2):
                with torch.no_grad():
                    y = model.decode(y)
                    vis_routine(metric, y, depth, p, depth_map[0])

                    if is_gradio:
                        yield torch_samps_to_imgs(
                            y
                        )[0], f"Progress: {pbar.n}/{pbar.total} \nAfter the generation is complete, the video results will be displayed below.", gr.update(
                            value=None)

            metric.step()
            pbar.update()

            pbar.set_description(p)
            hbeat.beat()

        metric.put_artifact("ckpt", ".pt", "",
                            lambda fn: torch.save(vox.state_dict(), fn))

        with EventStorage("result"):
            evaluate(model, vox, poser)

        if is_gradio:
            yield gr.update(
                visible=True
            ), f"Generation complete. Please check the video below. \nThe result files and logs are located at {exp_instance_dir}", gr.update(
                value=os.path.join(exp_instance_dir,
                                   '3d/result_10000/video/step_100_.mp4'))
        else:
            yield None

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
        y = score_model.decode(y)
        vis_routine(metric, y, depth, "", None)

        metric.step()
        hbeat.beat()

    metric.flush_history()

    metric.put_artifact(
        "video", ".mp4", "",
        lambda fn: stitch_vis(fn,
                              read_stats(metric.output_dir, "img")[1]))

    metric.step()


def render_one_view(vox, aabb, H, W, K, pose, return_w=False):
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)

    ro, rd, t_min, t_max = scene_box_filter_(ro, rd, aabb)

    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)
    rgbs, depth, weights = render_ray_bundle(vox, ro, rd, t_min, t_max)

    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H, w=W)
    if return_w:
        return rgbs, depth, weights
    else:
        return rgbs, depth


def scene_box_filter_(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth, prompt, depth_map):
    pane = nerf_vis(y, depth, final_H=256)
    im = torch_samps_to_imgs(y)[0]

    depth = depth.cpu().numpy()
    metric.put_artifact("view", ".png", "", lambda fn: imwrite(fn, pane))
    metric.put_artifact("img", ".png", prompt, lambda fn: imwrite(fn, im))
    if depth_map != None:
        metric.put_artifact("PC_depth", ".png", prompt,
                            lambda fn: imwrite(fn,
                                               depth_map.cpu().squeeze()))
    metric.put_artifact("depth", ".npy", "", lambda fn: np.save(fn, depth))


if __name__ == "__main__":
    dispatch(SJC_3DFuse)
