# Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation
<a href="https://arxiv.org/abs/2303.07937"><img src="https://img.shields.io/badge/arXiv-2303.07937-%23B31B1B"></a>
<a href="https://ku-cvlab.github.io/3DFuse/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<a href="https://huggingface.co/spaces/jyseo/3DFuse"><img src="https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565"></a>
<br>

<p align="center">
<img src="imgs/1.gif" width="40%">
<img src="imgs/2.gif" width="40%">
<img src="imgs/3.gif" width="40%">
<img src="imgs/4.gif" width="40%">
</p>
This is official implementation of the paper "Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation". The last column of each example is our result.

## ⚡️News
**❗️2023.04.10:**  We've opened the [**HuggingFace Demo**](https://huggingface.co/spaces/jyseo/3DFuse)! Also, we fixed minor issues, including the seed not being fixed.

**❗️2023.03.31:**  We found that we typed an incorrect version of the model for point cloud inference. **The fixed commit produces much better results.**

## Introduction
<center>
<img src="https://ku-cvlab.github.io/3DFuse/imgs/3dfuse.png" width="100%" height="100%"> 
</center>

We introduce 3DFuse, a novel framework that incorporates 3D awareness into pretrained 2D diffusion models, enhancing the robustness and 3D consistency of score distillation-based methods. For more details, please visit our [project page](https://ku-cvlab.github.io/3DFuse/)!

## 🔥TODO
- [x] 3D Generation/Gradio Demo Code
- [x] HuggingFace🤗 Demo Release
- [ ] Colab Demo Release
- [ ] Mesh Converting Code

## Installation
Please follow [installation](INSTALL.md).

## Interactive Gradio App
### for Text-to-3D / Image-to-3D
Enter your own prompt and enjoy! With this gradio app, you can preview the point cloud before 3D generation and determine the desired shape.
```
python gradio_app.py
# or python gradio_app.py --share
```
<img src='imgs/ex1.png'>
<img src='imgs/ex2.png'>

## Text-to-3D Generation
After modifying the `run.sh` file with the desired prompt and hyperparameters, please execute the following command:
```
sh run.sh
```

## Acknowledgement
We would like to acknowledge the contributions of public projects, including [SJC](https://github.com/pals-ttic/sjc/) and [ControlNet](https://github.com/lllyasviel/ControlNet) whose code has been utilized in this repository.

## Citation
```
@article{seo2023let,
  title={Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation},
  author={Seo, Junyoung and Jang, Wooseok and Kwak, Min-Seop and Ko, Jaehoon and Kim, Hyeonsu and Kim, Junho and Kim, Jin-Hwa and Lee, Jiyoung and Kim, Seungryong},
  journal={arXiv preprint arXiv:2303.07937},
  year={2023}
}
```
