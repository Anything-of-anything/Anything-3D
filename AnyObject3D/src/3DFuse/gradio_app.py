import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

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


class Intermediate:

    def __init__(self):
        self.images = None
        self.points = None
        self.is_generating = False


def gen_3d(model, intermediate, prompt, keyword, seed, ti_step, pt_step):
    intermediate.is_generating = True
    images, points = intermediate.images, intermediate.points
    if images is None or points is None:
        raise gr.Error("Please generate point cloud first")
    del model

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

    yield from model.run_gradio(points, exp_instance_dir)

    intermediate.is_generating = False


def gen_pc_from_prompt(intermediate, num_initial_image, prompt, keyword, type,
                       bg_preprocess, seed):

    seed_everything(seed=seed)
    if keyword not in prompt:
        raise gr.Error("Prompt should contain keyword!")
    elif " " in keyword:
        raise gr.Error("Keyword should be one word!")

    images = gen_init(num_initial_image=num_initial_image,
                      prompt=prompt,
                      seed=seed,
                      type=type,
                      bg_preprocess=bg_preprocess)
    points = point_e_gradio(images[0], 'cuda')

    intermediate.images = images
    intermediate.points = points

    coords = np.array(points.coords)
    trace = go.Scatter3d(x=coords[:, 0],
                         y=coords[:, 1],
                         z=coords[:, 2],
                         mode='markers',
                         marker=dict(size=2))

    layout = go.Layout(scene=dict(
        xaxis=dict(title="",
                   showgrid=False,
                   zeroline=False,
                   showline=False,
                   ticks='',
                   showticklabels=False),
        yaxis=dict(title="",
                   showgrid=False,
                   zeroline=False,
                   showline=False,
                   ticks='',
                   showticklabels=False),
        zaxis=dict(title="",
                   showgrid=False,
                   zeroline=False,
                   showline=False,
                   ticks='',
                   showticklabels=False),
    ),
                       margin=dict(l=0, r=0, b=0, t=0),
                       showlegend=False)

    fig = go.Figure(data=[trace], layout=layout)

    return images[0], fig


def gen_pc_from_image(intermediate, image, prompt, keyword, bg_preprocess,
                      seed):

    seed_everything(seed=seed)
    if keyword not in prompt:
        raise gr.Error("Prompt should contain keyword!")
    elif " " in keyword:
        raise gr.Error("Keyword should be one word!")

    if bg_preprocess:
        # from carvekit.api.high import HiInterface
        # interface = HiInterface(object_type="object",
        #                 batch_size_seg=5,
        #                 batch_size_matting=1,
        #                 device='cuda' if torch.cuda.is_available() else 'cpu',
        #                 seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
        #                 matting_mask_size=2048,
        #                 trimap_prob_threshold=231,
        #                 trimap_dilation=30,
        #                 trimap_erosion_iters=5,
        #                 fp16=False)

        # img_without_background = interface([image])
        # mask = np.array(img_without_background[0]) > 127

        # read mask from local_dir
        imgname = keyword.lower()
        mask = cv2.imread(f'./img_masks/{imgname}_mask.png')
        mask = (mask > 0)[:, :, 0]

        image = np.array(image)
        image[~mask] = [255., 255., 255.]
        # image[~mask] = 255.
        image = Image.fromarray(np.array(image))

    points = point_e_gradio(image, 'cuda')

    intermediate.images = [image]
    intermediate.points = points

    coords = np.array(points.coords)
    trace = go.Scatter3d(x=coords[:, 0],
                         y=coords[:, 1],
                         z=coords[:, 2],
                         mode='markers',
                         marker=dict(size=2))

    layout = go.Layout(scene=dict(
        xaxis=dict(title="",
                   showgrid=False,
                   zeroline=False,
                   showline=False,
                   ticks='',
                   showticklabels=False),
        yaxis=dict(title="",
                   showgrid=False,
                   zeroline=False,
                   showline=False,
                   ticks='',
                   showticklabels=False),
        zaxis=dict(title="",
                   showgrid=False,
                   zeroline=False,
                   showline=False,
                   ticks='',
                   showticklabels=False),
    ),
                       margin=dict(l=0, r=0, b=0, t=0),
                       showlegend=False)

    fig = go.Figure(data=[trace], layout=layout)

    return image, fig


def gen_init(num_initial_image,
             prompt,
             seed,
             type="Karlo",
             bg_preprocess=False):

    pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16) if type=="Karlo (Recommended)" \
        else DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to('cuda')

    view_prompt = [
        "front view of ", "overhead view of ", "side view of ", "back view of "
    ]

    if bg_preprocess:
        from carvekit.api.high import HiInterface
        interface = HiInterface(
            object_type="object",
            batch_size_seg=5,
            batch_size_matting=1,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=False)

    images = []
    generator = torch.Generator(device='cuda').manual_seed(seed)
    for i in range(num_initial_image):
        t = ", white background" if bg_preprocess else ", white background"
        if i == 0:
            prompt_ = f"{view_prompt[i%4]}{prompt}{t}"
        else:
            prompt_ = f"{view_prompt[i%4]}{prompt}"

        image = pipe(prompt_, generator=generator).images[0]

        if bg_preprocess:
            # motivated by NeuralLift-360 (removing bg)
            # NOTE: This option was added during the code orgranization process.
            # The results reported in the paper were obtained with [bg_preprocess: False] setting.
            img_without_background = interface([image])
            mask = np.array(img_without_background[0]) > 127
            image = np.array(image)
            image[~mask] = [255., 255., 255.]
            image = Image.fromarray(np.array(image))
        images.append(image)

    return images


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true', help="public url")
    args = parser.parse_args()

    model = None
    intermediate = Intermediate()
    demo = gr.Blocks(title="3DFuse Interactive Demo")

    with demo:
        gr.Markdown("# 3DFuse Interactive Demo")
        gr.Markdown(
            "### Official Implementation of the paper \"Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation\""
        )
        gr.Markdown(
            "Enter your own prompt and enjoy! With this demo, you can preview the point cloud before 3D generation and determine the desired shape."
        )

        with gr.Row():
            with gr.Column(scale=1., variant='panel'):

                with gr.Tab("Text to 3D"):
                    prompt_input = gr.Textbox(label="Prompt",
                                              value="a comfortable bed",
                                              interactive=True)
                    word_input = gr.Textbox(label="Keyword for initialization",
                                            value="bed",
                                            interactive=True)
                    semantic_model_choice = gr.Radio(
                        ["Karlo (Recommended)", "Stable Diffusion"],
                        label="Backbone for initial image generation",
                        value="Karlo (Recommended)",
                        interactive=True)
                    seed = gr.Slider(label="Seed",
                                     minimum=0,
                                     maximum=2100000000,
                                     step=1,
                                     randomize=True)
                    preprocess_choice = gr.Checkbox(
                        True,
                        label=
                        "Preprocess intially-generated image by removing background",
                        interactive=True)
                    with gr.Accordion("Advanced Options", open=False):
                        opt_step = gr.Slider(
                            0,
                            1000,
                            value=500,
                            step=1,
                            label='Number of text embedding optimization step')
                        pivot_step = gr.Slider(
                            0,
                            1000,
                            value=500,
                            step=1,
                            label='Number of pivotal tuning step for LoRA')
                    with gr.Row():
                        button_gen_pc = gr.Button("Generate Point Cloud",
                                                  interactive=True,
                                                  variant='secondary')
                        button_gen_3d = gr.Button("Generate 3D",
                                                  interactive=True,
                                                  variant='primary')

                with gr.Tab("Image to 3D"):
                    image_input = gr.Image(source='upload',
                                           type="pil",
                                           interactive=True)
                    prompt_input_2 = gr.Textbox(label="Prompt",
                                                value="a dog",
                                                interactive=True)
                    word_input_2 = gr.Textbox(
                        label="Keyword for initialization",
                        value="dog",
                        interactive=True)
                    seed_2 = gr.Slider(label="Seed",
                                       minimum=0,
                                       maximum=2100000000,
                                       step=1,
                                       randomize=True)
                    preprocess_choice_2 = gr.Checkbox(
                        True,
                        label=
                        "Preprocess intially-generated image by removing background",
                        interactive=False)
                    with gr.Accordion("Advanced Options", open=False):
                        opt_step_2 = gr.Slider(
                            0,
                            1000,
                            value=500,
                            step=1,
                            label='Number of text embedding optimization step')
                        pivot_step_2 = gr.Slider(
                            0,
                            1000,
                            value=500,
                            step=1,
                            label='Number of pivotal tuning step for LoRA')
                    with gr.Row():
                        button_gen_pc_2 = gr.Button("Generate Point Cloud",
                                                    variant='secondary')
                        button_gen_3d_2 = gr.Button("Generate 3D",
                                                    variant='primary')
                    gr.Markdown(
                        "Note: A photo showing the entire object in a front view is recommended. Also, our framework is not designed with the goal of single shot reconstruction, so it may be difficult to reflect the details of the given image."
                    )

                with gr.Row(scale=1.):
                    with gr.Column(scale=1.):
                        pc_plot = gr.Plot(label="Inferred point cloud")
                    with gr.Column(scale=1.):
                        init_output = gr.Image(label='Generated initial image',
                                               interactive=False)
                        # init_output.style(grid=1)

            with gr.Column(scale=1., variant='panel'):
                with gr.Row():
                    with gr.Column(scale=1.):
                        intermediate_output = gr.Image(
                            label="Intermediate Output", interactive=False)
                    with gr.Column(scale=1.):
                        logs = gr.Textbox(label="logs",
                                          lines=8,
                                          max_lines=8,
                                          interactive=False)
                with gr.Row():
                    video_result = gr.Video(
                        label="Video result for generated 3D",
                        interactive=False)

        gr.Markdown(
            "Note: Keyword is used for Textual Inversion. Please choose one important noun included in the prompt. This demo may be slower than directly running run_3DFuse.py."
        )

        # functions

        button_gen_pc.click(fn=partial(gen_pc_from_prompt,intermediate,4), inputs=[prompt_input, word_input, semantic_model_choice, \
            preprocess_choice, seed], outputs=[init_output, pc_plot])
        button_gen_3d.click(fn=partial(gen_3d,model,intermediate), inputs=[prompt_input, word_input, seed, opt_step, pivot_step], \
            outputs=[intermediate_output,logs,video_result])

        button_gen_pc_2.click(fn=partial(gen_pc_from_image,intermediate), inputs=[image_input, prompt_input_2, word_input_2, \
            preprocess_choice_2, seed_2], outputs=[init_output, pc_plot])
        button_gen_3d_2.click(fn=partial(gen_3d,model,intermediate), inputs=[prompt_input_2, word_input_2, seed_2, opt_step_2, pivot_step_2], \
            outputs=[intermediate_output,logs,video_result])

    demo.queue(concurrency_count=1)
    demo.launch(share=args.share)
