import os
import torch
import cv2
import argparse
import gradio as gr
from functools import partial
from my.config import BaseConf, dispatch_gradio
from run_3DFuse import SJC_3DFuse
import numpy as np
from PIL import Image
from pc_project import point_e
from diffusers import UnCLIPPipeline, DiffusionPipeline
from pc_project import point_e_gradio
import numpy as np
import plotly.graph_objs as go
from my.utils.seed import seed_everything
import os

def gen_pc_from_image(imgname, prompt, keyword, seed, bg_preprocess=True):

    seed_everything(seed=seed)
    if keyword not in prompt:
        raise gr.Error("Prompt should contain keyword!")
    elif " " in keyword:
        raise gr.Error("Keyword should be one word!")

    image = cv2.cvtColor(cv2.imread(f'./images/{imgname}_patch.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32)
    if bg_preprocess:
        mask = cv2.imread(f'./images/{imgname}_mask.png')
        mask = (mask > 0)[:, :, 0]
        image = np.array(image)
        image[~mask] = [255., 255., 255.]
        image = Image.fromarray(image.astype(np.uint8))

    points = point_e_gradio(image, 'cuda')

    images = [image]
    points = points
    return images, points


def gen_3d(images, points, prompt, keyword, seed, ti_step=500, pt_step=500):
    if images is None or points is None:
        raise gr.Error("Please generate point cloud first")
    seed_everything(seed)
    i = 0
    exp_instance_dir = os.path.join("./results",
                                    keyword + "_" + str(i).zfill(4))
    while os.path.exists(exp_instance_dir):
        i += 1
        exp_instance_dir = os.path.join("./results",
                                        keyword + "_" + str(i).zfill(4))

    initial_images_dir = os.path.join(exp_instance_dir, 'initial_image')
    os.makedirs(initial_images_dir, exist_ok=True)

    for idx, img in enumerate(images):
        img.save(os.path.join(initial_images_dir, f"instance{idx}.png"))

    model = dispatch_gradio(SJC_3DFuse, prompt, keyword, ti_step, pt_step,
                            seed, exp_instance_dir)
    model.run_offline(points, exp_instance_dir)
