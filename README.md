![logo](logo.jpg)
<!-- # Anything-3D -->

We plan to create a very interesting demo by combining [Segment Anything](https://github.com/facebookresearch/segment-anything) and a series of 3D models! Right now, this is just a simple small project. We will continue to improve it and create more interesting demos. 

- [Anything-3DNovel-View](#anything-3dnovel-view)
- [Any-3DFace](#any-3dface)
- [:cupid: Acknowledgements](#cupid-acknowledgements)
- [Citation](#citation)
## Anything-3DNovel-View
- SAM + [Zero 1-to-3](https://github.com/cvlab-columbia/zero123)

![1](novel-view/assets/1.jpeg)
![2](novel-view/assets/2.jpeg)
![3](novel-view/assets/3.jpeg)

## Any-3DFace
- SAM + [HRN](https://younglbw.github.io/HRN-homepage/)

| Segmentation | Result|
| --- | ---|
| <img src="AnyFace3D/assets/celebrity_selfie/mask_1.jpg" width="2000"> | ![1](AnyFace3D/assets/celebrity_selfie/1.gif) |
| <img src="AnyFace3D/assets/celebrity_selfie/mask_2.jpg" width="2000">| ![3](AnyFace3D/assets/celebrity_selfie/2.gif) |
| <img src="AnyFace3D/assets/celebrity_selfie/mask_3.jpg" width="2000">| ![3](AnyFace3D/assets/celebrity_selfie/3.gif) |



## :cupid: Acknowledgements
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Zero 1-to-3](https://github.com/cvlab-columbia/zero123)
- [HRN](https://younglbw.github.io/HRN-homepage/)

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{kirillov2023segany,
    title={Segment Anything}, 
    author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
    journal={arXiv:2304.02643},
    year={2023}
}
@misc{liu2023zero1to3,
    title={Zero-1-to-3: Zero-shot One Image to 3D Object}, 
    author={Ruoshi Liu and Rundi Wu and Basile Van Hoorick and Pavel Tokmakov and Sergey Zakharov and Carl Vondrick},
    year={2023},
    eprint={2303.11328},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
@inproceedings{Lei2023AHR,
    title={A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images},
    author={Biwen Lei and Jianqiang Ren and Mengyang Feng and Miaomiao Cui and Xuansong Xie},
    year={2023}
}
```
