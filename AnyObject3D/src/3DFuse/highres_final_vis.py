import numpy as np
import torch
from einops import rearrange

from voxnerf.render import subpixel_rays_from_img

from run_sjc import (
    SJC, ScoreAdapter, StableDiffusion,
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak, get_event_storage, get_heartbeat, optional_load_config, read_stats,
    vis_routine, stitch_vis, latest_ckpt,
    scene_box_filter, render_ray_bundle, as_torch_tsrs,
    device_glb
)


# the SD deocder is very memory hungry; the latent image cannot be too large
# for a graphics card with < 12 GB memory, set this to 128; quality already good
# if your card has 12 to 24 GB memory, you can set this to 200;
# but visually it won't help beyond a certain point. Our teaser is done with 128.
decoder_bottleneck_hw = 128


def final_vis():
    cfg = optional_load_config(fname="full_config.yml")
    assert len(cfg) > 0, "can't find cfg file"
    mod = SJC(**cfg)

    family = cfg.pop("family")
    model: ScoreAdapter = getattr(mod, family).make()
    vox = mod.vox.make()
    poser = mod.pose.make()

    pbar = tqdm(range(1))

    with EventStorage(), HeartBeat(pbar):
        ckpt_fname = "/home/cvlab07/project/wooseok/DiffNeRF/sjc/corgi_no_over_const_intrinsic/control_depth_100/ckpt/step_10000.pt"
        state = torch.load(ckpt_fname, map_location="cpu")
        vox.load_state_dict(state)
        vox.to(device_glb)

        with EventStorage("highres"):
            # what dominates the speed is NOT the factor here.
            # you can try from 2 to 8, and the speed is about the same.
            # the dominating factor in the pipeline I believe is the SD decoder.
            evaluate(model, vox, poser, n_frames=200, factor=4)


@torch.no_grad()
def evaluate(score_model, vox, poser, n_frames=200, factor=4):
    H, W = poser.H, poser.W
    vox.eval()
    K, poses = poser.sample_test(n_frames)
    del n_frames
    poses = poses[60:]  # skip the full overhead view; not interesting

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
        y, depth = highres_render_one_view(vox, aabb, H, W, K, pose, f=factor)
        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        vis_routine(metric, y, depth)

        metric.step()
        hbeat.beat()

    metric.flush_history()

    metric.put_artifact(
        "movie_im_and_depth", ".mp4",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "view")[1])
    )

    metric.put_artifact(
        "movie_im_only", ".mp4",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "img")[1])
    )

    metric.step()


def highres_render_one_view(vox, aabb, H, W, K, pose, f=4):
    bs = 4096

    ro, rd = subpixel_rays_from_img(H, W, K, pose, f=f)
    ro, rd, t_min, t_max = scene_box_filter(ro, rd, aabb)
    n = len(ro)
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)

    rgbs = torch.zeros(n, 4, device=vox.device)
    depth = torch.zeros(n, 1, device=vox.device)

    with torch.no_grad():
        for i in range(int(np.ceil(n / bs))):
            s = i * bs
            e = min(n, s + bs)
            _rgbs, _depth, _ = render_ray_bundle(
                vox, ro[s:e], rd[s:e], t_min[s:e], t_max[s:e]
            )
            rgbs[s:e] = _rgbs
            depth[s:e] = _depth

    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H*f, w=W*f)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H*f, w=W*f)
    rgbs = torch.nn.functional.interpolate(
        rgbs, (decoder_bottleneck_hw, decoder_bottleneck_hw),
        mode='bilinear', antialias=True
    )
    return rgbs, depth


if __name__ == "__main__":
    final_vis()
