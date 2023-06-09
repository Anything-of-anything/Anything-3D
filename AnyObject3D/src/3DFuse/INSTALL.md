# Installation
An example of installation is shown below:
```
git clone https://github.com/KU-CVLAB/3DFuse
cd 3DFuse
conda create -n 3DFuse python=3.8
conda activate 3DFuse
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install pytorch3d -c pytorch3d
```
If PyTorch3D installation fails, please try
```
pip install git+https://github.com/facebookresearch/pytorch3d@c8af1c45ca9f4fdd4e59b49172ca74983ff3147a#egg=pytorch3d
```
or follow the instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

And then,
```
pip install -r requirements.txt
```

## Pretrained Weights
The weights required for the operation of our framework can be downloaded from [our HuggingFace page](https://huggingface.co/jyseo/3DFuse_weights). Please modify the `env.json` file with the path of the weight file afterwards.
```
mkdir weights
cd weights
wget https://huggingface.co/jyseo/3DFuse_weights/resolve/main/models/3DFuse_sparse_depth_injector.ckpt
```