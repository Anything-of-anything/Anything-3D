import numpy as np
from numpy import sin, cos
from math import pi as π
from my3d import camera_pose
from my.config import BaseConf


def get_K(H, W, FoV_x):
    FoV_x = FoV_x / 180 * π  # to rad
    f = 1 / np.tan(FoV_x / 2) * (W / 2)

    K = np.array([
        [f, 0, -(W/2 - 0.5)],
        [0, -f, -(H/2 - 0.5)],
        [0, 0, -1]
    ])

    
    return K


SIDEVIEW_PROMPTS = [
    "front view of", "side view of", "backside view of", "side view of"
]

TOPVIEW_PROMPT = "overhead view of"


def train_eye_with_prompts(r, n):
    hs = np.random.rand(n) * 360
    vs = np.random.rand(n) * np.deg2rad(100)
    vs = np.clip(vs, 1e-2, π-1e-2)

    prompts = []
    v_thresh = np.deg2rad(30)
    for i in range(n):
        _p = ""
        if vs[i] < v_thresh:
            _p = TOPVIEW_PROMPT
        else:
            _a = hs[i]
            _a = (_a + 45) % 360
            _quad = int(_a // 90)
            _p = SIDEVIEW_PROMPTS[_quad]
        prompts.append(_p)

    θ = np.deg2rad(hs)
    # φ = v
    φ = np.arccos(1 - 2 * (vs / π))
    
    horz = hs
    elev = np.rad2deg(π / 2 - φ)

    eyes = np.zeros((n, 3))

    eyes[:, 0] = r * sin(φ) * cos(π-θ)  # x
    eyes[:, 2] = r * sin(φ) * sin(π-θ)  # z
    eyes[:, 1] = r * cos(φ)  # y
    
    return eyes, prompts, horz, elev


def spiral_poses(
    radius, height,
    num_steps=20, num_rounds=1,
    center=np.array([0, 0, 0]), up=np.array([0, 1, 0]),
):
    eyes = []
    for i in range(num_steps):
        ratio = (i + 1) / num_steps
        Δz = height * (1 - ratio)

        θ = ratio * (360 * num_rounds)
        θ = θ / 180 * π
        # _r = max(radius * ratio, 0.5)
        _r = max(radius * sin(ratio * π / 2), 0.5)
        Δx, Δy = _r * np.array([np.cos(θ), np.sin(θ)])
        eyes.append(center + [Δx, Δz, Δy])

    poses = [
        camera_pose(e, center - e, up) for e in eyes
    ]
    return poses

def circular_poses(
    radius, height,
    num_steps=36, num_rounds=1,
    center=np.array([0, 0, 0]), up=np.array([0, 1, 0]),
):
    eyes = []
    horz = []
    for i in range(num_steps):
        ratio = (i + 1) / num_steps
        Δz = radius * (np.cos(np.deg2rad(60)))

        θ = ratio * (360 * num_rounds)
        θ = θ / 180 * π
        # _r = max(radius * ratio, 0.5)
        _r = max(radius * sin(ratio * π / 2), 0.5)
        Δx, Δy = radius * np.array([np.sin(np.deg2rad(60))*np.cos(θ), np.sin(np.deg2rad(60))*np.sin(θ)])
        eyes.append(center + [Δx, Δz, Δy])
        horz.append(np.rad2deg(θ))
    poses = [
        camera_pose(e, center - e, up) for e in eyes
    ]
    horz=np.array(horz)
    return poses

class PoseConfig(BaseConf):
    rend_hw: int = 64
    FoV: float = 60.0
    R: float = 1.5

    def make(self):
        cfgs = self.dict()
        hw = cfgs.pop("rend_hw")
        cfgs["H"] = hw
        cfgs["W"] = hw
        return Poser(**cfgs)


class Poser():
    def __init__(self, H, W, FoV, R):
        self.H, self.W = H, W
        self.R = R
        self.K = get_K(H, W, FoV)
        self.FoV = FoV

    def sample_train(self, n, device):
        eyes, prompts, horz, elev = train_eye_with_prompts(r=self.R, n=n)
        up = np.array([0, 1, 0])

        
        poses = [
            camera_pose(e, -e, up) for e in eyes
        ]
        poses = np.stack(poses, 0)
        

        
        FoV = np.random.rand(n) * 30 + 40


        random_Ks = [
            get_K(self.H, self.W, FoV[i])
            for i in range(len(poses))

        ]


        angles_list = []
        
        for horizontal, elevation, fov in zip(horz, elev, FoV):
            angles_list.append([horizontal, elevation, fov])
        
        return random_Ks, poses, prompts, angles_list

    def sample_test(self, n):
        poses = circular_poses(self.R, self.R, n, num_rounds=3)
        poses = np.stack(poses, axis=0)
        return self.K, poses