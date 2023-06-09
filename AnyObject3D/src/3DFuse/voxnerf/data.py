from pathlib import Path
import json
import numpy as np
import imageio
from .utils import blend_rgba


def load_blender(split, scene="lego", half_res=False):
    assert split in ("train", "val", "test")

    env_fname = Path(__file__).resolve().parents[1] / "env.json"
    with env_fname.open("r") as f:
        root = json.load(f)['data_root']
    root = Path(root) / scene

    with open(root / f'transforms_{split}.json', "r") as f:
        meta = json.load(f)

    imgs, poses = [], []

    for frame in meta['frames']:
        file_name = root / f"{frame['file_path']}.png"
        im = imageio.imread(file_name)
        c2w = frame['transform_matrix']

        imgs.append(im)
        poses.append(c2w)

    imgs = (np.array(imgs) / 255.).astype(np.float32)  # (RGBA) imgs
    imgs = blend_rgba(imgs)
    poses = np.array(poses).astype(float)
    # print(imgs.shape)
    H, W = imgs[0].shape[:2]
    W = 64
    H = 64
    camera_angle_x = float(meta['camera_angle_x'])
    f = 1 / np.tan(camera_angle_x / 2) * (W / 2)

    if half_res:
        raise NotImplementedError()
    
    K = np.array([
        [f, 0, -(W/2 - 0.5)],
        [0, -f, -(H/2 - 0.5)],
        [0, 0, -1]
    ])  # note OpenGL -ve z convention;

    return imgs, K, poses
