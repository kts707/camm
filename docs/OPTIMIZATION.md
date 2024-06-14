## iiwa dataset

Note that for eagle dataset, we use ground-truth camera poses under raw/iiwa/

### Initial optimization stage

```
# define sequence name
seqname=iiwa

# user defined extra_tag to differentiate between different experiments
extra_tag=iiwa-test1

# opt config file
flagfile=opt_configs/iiwa/init/iiwa-dino

# optimization
bash scripts/template-known-cam.sh 0 10001 $flagfile $extra_tag

# argv[1]: gpu id
# args[2]: port for distributed training
# args[3]: opt config file
# args[4]: extra_tag


# visualize the surface reconstruction results
bash scripts/render_mgpu.sh 0 $seqname logdir/$extra_tag/$seqname-ft2/params_latest.pth "1" 256

# argv[1]: gpu id
# args[2]: sequence name
# argv[3]: weights path
# argv[4]: video ids to visualize
# argv[5]: resolution of running marching cubes
```


### Get initial estimate of the kinematic chain

Go to RigNet folder:

```
cd third_party/RigNet
conda activate rignet


# extract and save the kinematic chain
python extract_skel.py --mesh_path ../../logdir/$extra_tag/$seqname-ft2-rendering/mesh-rest.obj --mesh_name iiwa --output_path iiwa_joints.pkl --symm

# arguments for extract_skel.py:
# --mesh_path: the path to the canonical mesh (.obj file)
# --mesh_name: user defined mesh name for preprocessing (preprocessed mesh will be saved as {mesh_name}_remesh.obj)
# --output_path: output path to save the kinematic chain .pkl file
# --symm: whether to extract symmetric kinematic chain (optional)


# switch back to the default conda environment and default directory
conda activate camm
cd ..;cd ..
mv third_party/RigNet/iiwa_joints.pkl iiwa_joints.pkl
```

(optional) Modify the .pkl file's path in draw_kinematic_chain.py to visualize the kinematic chain
```
python draw_kinematic_chain.py
```

If the kinematic chain does not look reasonable, it's possible to tune the bandwidth and threshold [here](https://github.com/kts707/camm/blob/main/third_party/RigNet/extract_skel.py#L457) to get a better kinematic chain. We suggest the users to tune it to get a good kinematic chain before starting the kinematic chain aware optimization.

(optional) To directly use the kinematic chain initialization and visualize the results, simply run:

```
bash scripts/render_mgpu_skel.sh 0 $seqname logdir/$extra_tag/$seqname-ft2/params_latest.pth "1" 256 iiwa-joints.pkl

# argv[1]: gpu id
# args[2]: sequence name
# argv[3]: weights path
# argv[4]: video ids to visualize
# argv[5]: resolution of running marching cubes
# args[6]: kinematic chain .pkl file
```


### Kinematic Chain Aware Optimization

Assuming a good kinematic chain is obtained from RigNet (.pkl file)

#### Optimization
```
# define kinematic chain .pkl file
kinematic-chain=iiwa_joints.pkl

flagfile=opt_configs/iiwa/skel/update-all

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


## Eagle dataset

Note that for eagle dataset, we use ground-truth camera poses under database/DAVIS/Camera

### Initial optimization stage

```
# define sequence name
seqname=a-eagle

# user defined extra_tag to differentiate between different experiments
extra_tag=eagle-test1

# opt config file
flagfile=opt_configs/eagle/init/eagle-dino

# optimization
bash scripts/template-known-cam.sh 0 10001 $flagfile $extra_tag

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


### Get initial estimate of the kinematic chain

Go to RigNet folder:

```
cd third_party/RigNet
conda activate rignet


# extract and save the kinematic chain
python extract_skel.py --mesh_path logdir/$extra_tag/$seqname-ft2-rendering/mesh-rest.obj --mesh_name eagle --output_path eagle_joints.pkl --symm

# arguments for extract_skel.py:
# --mesh_path: the path to the canonical mesh (.obj file)
# --mesh_name: user defined mesh name for preprocessing (preprocessed mesh will be saved as {mesh_name}_remesh.obj)
# --output_path: output path to save the kinematic chain .pkl file
# --symm: whether to extract symmetric kinematic chain (optional)


# switch back to the default conda environment and default directory
conda activate camm
cd ..;cd ..
mv third_party/RigNet/eagle_joints.pkl eagle_joints.pkl
```

(optional) Modify the .pkl file's path in draw_kinematic_chain.py to visualize the kinematic chain
```
python draw_kinematic_chain.py
```

If the kinematic chain does not look reasonable, it's possible to tune the bandwidth and threshold [here](https://github.com/kts707/camm/blob/main/third_party/RigNet/extract_skel.py#L457) to get a better kinematic chain. We suggest the users to tune it to get a good kinematic chain before starting the kinematic chain aware optimization.

(optional) To directly use the kinematic chain initialization and visualize the results, simply run:

```
bash scripts/render_mgpu_skel.sh 0 $seqname logdir/$extra_tag/$seqname-ft2/params_latest.pth "0" 256 eagle-joints.pkl

# argv[1]: gpu id
# args[2]: sequence name
# argv[3]: weights path
# argv[4]: video ids to visualize
# argv[5]: resolution of running marching cubes
# args[6]: kinematic chain .pkl file
```


### Kinematic Chain Aware Optimization

Assuming a good kinematic chain is obtained from RigNet (.pkl file)

#### Optimization
```
# define kinematic chain .pkl file
kinematic-chain=eagle_joints.pkl

flagfile=opt_configs/eagle/skel/update-all

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