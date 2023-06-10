# Anything-3D: Towards Single-view Anything Reconstruction in the Wild**

This repository contains the official implementation of Anything-3D, a novel framework designed to address the challenges of 3D reconstruction from a single RGB image in unconstrained real-world scenarios. The diversity and complexity of objects and environments pose significant difficulties in accurately reconstructing 3D geometry from a single viewpoint. Anything-3D presents a systematic approach that combines visual-language models and the Segment-Anything object segmentation model to elevate objects to 3D, resulting in a reliable and versatile system for the single-view conditioned 3D reconstruction task.


## Preparation

Before running the code, make sure to install the required dependencies listed in the requirements.txt file.
Running Environment
   ```bash 
   # we use cuda-11.3 runtime.
   cd /path/to/Anything-3D/AnyObject3D/src
   pip install -r 3DFuse/requirements.txt
   pip install -r requirements.txt
   ```
Pretrained models 
   ```bash
   cd /path/to/Anything-3D/AnyObject3D/src
   mkdir weights && cd weights 
   wget https://huggingface.co/jyseo/3DFuse_weights/resolve/main/models/3DFuse_sparse_depth_injector.ckpt
   ```
## Get 3D Objects

Now you can have a try to recontruct a 3D object from a single view image in the wild.
```bash 
# rename image with your desired object name and move it into the folder src/images/ 
# a simple demo to get a 3d car
mv car.jpg images/
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD/3DFuse:$PYTHONPATH
python ./main.py --image car \ # the name of your input image
                --point_coords 1000 1000 \ # point prompt in segment anything
                --seed 302651 # random seed 
```
The reconstruction process takes about 90 minutes on single RTX3090. After reconstruction, you can find the 3D car in the `results` folder. 
```bash
# find the resulted multi-view video 
find ./results -name '*.mp4'
```
If your have any other problems, feel free to open an issue at this repo. 


## Acknowledgement
We express our gratitude to the exceptional project that inspired our code.
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)
- [3DFuse](https://github.com/KU-CVLAB/3DFuse)
- [Point-E](https://github.com/openai/point-e)
- [BLIP](https://github.com/salesforce/LAVIS)
