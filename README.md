# CAMM 
This repository contains the official implementation for the paper:

[**CAMM: Building Category-Agnostic and Animatable 3D Models from Monocular Videos**](https://camm3d.github.io/)

[Tianshu Kuai](https://tianshukuai.github.io/), [Akash Karthikeyan](https://aku02.github.io/), [Yash Kant](https://yashkant.github.io/), [Ashkan Mirzaei](https://scholar.google.com/citations?user=z8GwuTgAAAAJ&hl=en), [Igor Gilitschenski](https://www.gilitschenski.org/igor/)

[CVPR 2023 DynaVis Workshop](https://dynavis.github.io/2023/)

Please visit our [project page](https://camm3d.github.io/) for more qualitative results and a brief overview of our approach. Our iiwa robotic arm dataset can be accessed and downloaded from [here](https://drive.google.com/drive/folders/12cXPwA-hV4zjFhn_shvtnfxHDl5QxeUS?usp=sharing).

## Overview
- [Changelog](#changelog)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Optimization](#optimization)
- [Explicit Re-posing](#explicit-re-posing)
- [Quantitative Evaluation](#quantitative-evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## Changelog
[2023-04-14] `CAMM` is released.

## Installation

The code is tested in Python 3.9 with cuda 11.6 on a RTX 3090 GPU.
```
# clone repo
git clone --recursive https://github.com/kts707/camm
cd camm

# install conda environment
conda env create -f misc/camm.yml
conda activate camm

# install pytorch3d, kmeans-pytorch, and detectron2
pip install -e third_party/pytorch3d
pip install -e third_party/kmeans_pytorch
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# need to run this line only if running into CUBLAS_STATUS_NOT_SUPPORTED error (optional)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# install rignet's environment
conda deactivate
conda env create -f misc/rignet.yml
```

## Data Preparation
Please see [here](./docs/DATA.md) for detailed steps on data preparation for each dataset.

## Optimization

Our pipeline has two stages: the initial optimization stage and the kinematic chain aware optimization stage.

### 1. Initial optimization stage

We provide the instructions for optimization on AMA Human dataset here as an example. To run optimization for other datasets, simply change the sequence name
and extra tag, or see the examples [here](./docs/OPTIMIZATION.md).
```
# define sequence name
seqname=ama-female

# user defined extra_tag to differentiate between different experiments
extra_tag=ama-test1

# opt config file
flagfile=opt_configs/ama/init/ama-female-dp

# optimization
bash scripts/template.sh 0 10001 $flagfile $extra_tag

# argv[1]: gpu id
# args[2]: port for distributed training
# args[3]: opt config file
# args[4]: extra_tag


# visualize the surface reconstruction results
bash scripts/render_mgpu.sh 0 $seqname logdir/$extra_tag/$seqname-ft2/params_latest.pth "0" 256

# argv[1]: gpu id
# args[2]: sequence name
# argv[3]: weights path
# argv[4]: video ids to visualize
# argv[5]: resolution of running marching cubes
```


### 2. Get initial estimate of the kinematic chain

Go to RigNet folder:

```
cd third_party/RigNet
conda activate rignet


# extract and save the kinematic chain
python extract_skel.py --mesh_path ../../logdir/$extra_tag/$seqname-ft2-rendering/mesh-rest.obj --mesh_name ama --output_path ama_joints.pkl --symm

# arguments for extract_skel.py
# --mesh_path: the path to the canonical mesh (.obj file)
# --mesh_name: user defined mesh name for preprocessing (preprocessed mesh will be saved as {mesh_name}_remesh.obj)
# --output_path: output path to save the kinematic chain .pkl file
# --symm: whether to extract symmetric kinematic chain (optional)


# switch back to the default conda environment and default directory
conda activate camm
cd ..;cd ..
mv third_party/RigNet/ama_joints.pkl ama_joints.pkl
```

(optional) Modify the .pkl file's path in draw_kinematic_chain.py to visualize the kinematic chain:
```
python draw_kinematic_chain.py
```

If the kinematic chain does not look reasonable, it's possible to tune the bandwidth and threshold [here](https://github.com/kts707/camm/blob/main/third_party/RigNet/extract_skel.py#L457) to get a better kinematic chain. We suggest the users to tune it to get a good kinematic chain before starting the kinematic chain aware optimization.

(optional) To directly use the kinematic chain initialization and visualize the results, simply run:

```
bash scripts/render_mgpu_skel.sh 0 $seqname logdir/$extra_tag/$seqname-ft2/params_latest.pth "0" 256 ama-joints.pkl

# argv[1]: gpu id
# args[2]: sequence name
# argv[3]: weights path
# argv[4]: video ids to visualize
# argv[5]: resolution of running marching cubes
# args[6]: kinematic chain .pkl file
```


### 3. Kinematic chain aware optimization

Assuming a good kinematic chain is obtained from RigNet (.pkl file)

#### Optimization
```
# define kinematic chain .pkl file
kinematic-chain=ama_joints.pkl

flagfile=opt_configs/ama/skel/update-all

bash scripts/template-kinematic-chain.sh 0 10001 $flagfile $extra_tag $kinematic-chain

# argv[1]: gpu id
# args[2]: port for distributed training
# args[3]: opt config file
# args[4]: same extra_tag as before
# args[5]: kinematic chain .pkl file
```

#### Viusalizing Results
```
bash scripts/render_mgpu_skel.sh 0 $seqname logdir/$extra_tag/$seqname-skel/params_latest.pth "0" 256 $kinematic-chain

# argv[1]: gpu id
# args[2]: sequence name
# argv[3]: weights path
# argv[4]: video ids to visualize
# argv[5]: resolution of running marching cubes
# args[6]: kinematic chain .pkl file
```

## Explicit Re-posing
We provide an example of directly re-posing the learned kinematic chain and mesh for the AMA female [here](docs/REPOSING.md). 

Note that it is using our pre-trained checkpoint, so users can directly run it after installation and ANA data preparation (no training needed). 

## Quantitative Evaluation
Please follow the detailed instructions [here](docs/EVALUATION.md) to run quantitative evaluation for each dataset.

## Acknowledgement
Our code is mainly built based on [BANMo](https://github.com/facebookresearch/banmo). We thank the authors for sharing the code and for the help in explaining the code!

We also use the external repositories listed below in this project. A big thanks to them for their code!
- [RigNet](https://github.com/zhan-xu/RigNet)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [SoftRas](https://github.com/ShichenLiu/SoftRas)
- [Chamfer3D](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)
- [Nerf_pl](https://github.com/kwea123/nerf_pl)
- [VCN-robust](https://github.com/gengshan-y/rigidmask)


## Citation
If you find this project useful in your research, please consider citing:
```
@InProceedings{Kuai_2023_CVPR,
    author    = {Kuai, Tianshu and Karthikeyan, Akash and Kant, Yash and Mirzaei, Ashkan and Gilitschenski, Igor},
    title     = {CAMM: Building Category-Agnostic and Animatable 3D Models From Monocular Videos},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {6586-6596}
}
```

## License

Please see the [LICENSE](LICENSE) file. 
