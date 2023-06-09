import json
import numpy as np
from numpy.linalg import inv
from pathlib import Path
import imageio
import open3d as o3d

from hc3d.vis import CameraCone
from hc3d.render import compute_intrinsics, unproject
from hc3d.utils import batch_img_resize
from fabric.utils.seed import seed_everything


def get_K(H=500, W=500, fov=60):
    K = compute_intrinsics(W / H, fov, H)
    return K


def shoot_rays(K, pose):
    h = 200
    pixs = np.array([
        [10, h],
        [200, h],
        [400, h]
    ])
    pts = unproject(K, pixs, depth=1.0)
    pts = np.concatenate([
        pts,
        np.array([0, 0, 0, 1]).reshape(1, -1),
    ], axis=0)  # origin, followed by 4 img corners
    pts = pts @ pose.T
    pts = pts[:, :3]
    pts = pts.astype(np.float32)

    n = len(pixs)
    lines = np.array([
        [i, n] for i in range(n)
    ], dtype=np.int32)

    color = [1, 1, 0]
    colors = np.array([color] * len(lines), dtype=np.float32)

    lset = o3d.t.geometry.LineSet()
    lset.point['positions'] = pts
    lset.line['indices'] = lines
    lset.line['colors'] = colors

    return lset


def test_rays(H, W, K):
    xs, ys = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32), indexing='xy'
    )
    xys = np.stack([xs, ys], axis=-1)
    my_rays = unproject(K, xys.reshape(-1, 2))
    my_rays = my_rays.reshape(int(H), int(W), 4)[:, :, :3]
    return


def plot_inward_facing_views():
    # from run_sjc import get_train_poses
    from math import pi
    from pose import Poser
    H, W = 64, 64
    poser = Poser(H, W, FoV=60, R=4)
    # K, poses = poser.sample_test(100)
    K, poses, _ = poser.sample_train(1000)
    K = K[0]

    cam_locs = poses[:, :3, -1]
    # radius = np.linalg.norm(cam_locs, axis=1)
    # print(f"scene radius {radius}")

    # test_rays(H, W, K)

    # K = get_K(H, W, 50)
    # NeRF blender actually follows OpenGL camera convention (except top-left corner); nice
    # but its world coordinate is z up. I find it strange.

    def generate_cam(po, color, im=None):
        cone = CameraCone(K, po, W, H, scale=0.1,
                          top_left_corner=(0, 0), color=color)
        lset = cone.as_line_set()
        if im is None:
            return [lset]
        else:
            # o3d img tsr requires contiguous array
            im = np.ascontiguousarray(im)
            view_plane = cone.as_view_plane(im)
            return [lset, view_plane]

    cones = []

    for i in range(len(poses)):
        po = poses[i]
        geom = generate_cam(po, [1, 0, 0])
        cones.extend(geom)
        # rays = shoot_rays(K, po)
        # cones.extend([rays])

    o3d.visualization.draw(cones, show_skybox=False)


def blend_rgba(img):
    img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])  # blend A to RGB
    return img


def compare():
    import math
    import matplotlib.pyplot as plt

    vs = np.linspace(1e-5, math.pi - 1e-5, 500)
    phi = np.arccos(1 - 2 * (vs / math.pi))
    plt.plot(vs, phi)
    plt.show()


if __name__ == "__main__":
    seed_everything(0)
    plot_inward_facing_views()
    # compare()
