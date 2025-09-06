# Generalizable Human Gaussians from Single-View Image (ICLR 2025)

## [Project Page](https://jinnan-chen.github.io/projects/HGM/) | [Paper](https://arxiv.org/abs/2406.06050)

## Overview
We propose a single-view generalizable Human Gaussian Model (HGM), a diffusion-guided framework for 3D human modeling from a single image. Our approach enables high-quality human reconstruction that generalizes well to novel viewpoints.

## Installation

### Dependencies
```bash
pip install -r requirements.txt
```

### Install spconv
```bash
git clone https://github.com/traveller59/spconv --recursive
cd spconv
git checkout abf0acf30f5526ea93e687e3f424f62d9cd8313a
export CUDA_HOME="/usr/local/cuda-10.0"
python setup.py bdist_wheel
cd dist
pip install spconv-1.2.1-cp36-cp36m-linux_x86_64.whl
```

### Body Models
Download the following models:
- SMPL model from [SMPL website](https://smpl.is.tue.mpg.de/)
- SMPLX model from [SMPLX website](https://smpl-x.is.tue.mpg.de/)

## Dataset Preparation
You need to prepare your own dataset with multiview renderings:

Structure your dataset following this format:
```
dataset/
├── img/
│   ├── 0000_000/0.jpg     # (sample0-view0)
│   ├── 0000_001/0.jpg     # (sample0-view1)
│   ├── ...
│   └── camera_params.json
├── parm/
│   ├── 0000_000/0_extrinsic.npy    # (sample0-view0)
│   ├── 0000_001/0_extrinsic.npy    # (sample0-view1)
│   ├── ...
│   └── camera_params.json
├── mask/
│   ├── ...
│   └── ...
```

Adjust the camera system in `provider_objaverse.py` to fit your dataset's camera configuration.

## Training

The training code is located in `main.py`. 

We supervise gaussians from three different sources:
- Gaussians from UNet predictions
- Gaussians from the SMPL branch
- Merged Gaussians from both branches

## Infer

CUDA_VISIBLE_DEVICES=0 python3 infer.py big --resume ### --workspace ### --num_workers 1 --num_input_views 1 --mode '5'  --dataset ### --mode4 'att'  --mode5 'att'  --smplx

## Acknowledgements
Our code is based on:
- https://github.com/3DTopia/LGM.git
- https://github.com/sail-sg/GP-Nerf (SparseConv)

## Citation
If you find our code or paper helps, please consider citing:
```
@article{chen2024generalizable,
  title={Generalizable Human Gaussians from Single-View Image},
  author={Chen, Jinnan and Li, Chen and Zhang, Jianfeng and Zhu, Lingting and Huang, Buzhen and Chen, Hanlin and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2406.06050},
  year={2024}}
```
