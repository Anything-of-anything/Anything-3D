# AnyObject3D

## Preparation

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
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)
- [3DFuse](https://github.com/KU-CVLAB/3DFuse)
- [Point-E](https://github.com/openai/point-e)
- [BLIP](https://github.com/salesforce/LAVIS)