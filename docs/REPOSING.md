## Explicit Re-posing

We provide one example of direct pose manipulation by user defined transformation under example/ 

Note that running it does not require training beforehand, only installation and AMA data preparation are needed.

Please see Line 55 in [scripts/repose/repose_single.py](scripts/repose/repose_single.py) for an example on how to define custom rotations for desired kinematic chain links. Users can define whatever rotations they would like to apply to any kinematic chain link.

```
# simply re-posing
# re-posed kinematic chain and mesh are saved under example/output_single

bash scripts/render_repose_single.sh 0 example example/ama_joints.pkl

# argv[1]: gpu id
# args[2]: path to the directory that contains the model ckpt
# args[3]: path to the kinematic chain .pkl file
```

To make a video (100 frames) for the same defined re-posing to show movement, simply run the following script:
```
# save a 100 frames sequence
# re-posed kinematic chains and meshes are saved under example/output_sequence

bash scripts/render_repose_single.sh 0 example example/ama_joints.pkl

# argv[1]: gpu id
# args[2]: path to the directory that contains the model ckpt
# args[3]: path to the kinematic chain .pkl file
```

Use the following script to make a video for the saved sequence:
```
# generate a video for the kinematic chain movement
python scripts/visualize/render_repose.py --testdir example --meshdir example/output_sequence --first_idx 0 --last_idx 100 --draw_skel --vp 0

# generate a video for the mesh movement
python scripts/visualize/render_repose.py --testdir example --meshdir example/output_sequence --first_idx 0 --last_idx 100 --draw_mesh --vp 0
```