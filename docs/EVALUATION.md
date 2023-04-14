## Installation
Install chamfer3D
```
pip install -e third_party/chamfer3D/
```

## Quantitative Evaluations

### AMA-swing
```
bash scripts/eval/run_eval_ama_swing.sh 0 logdir/$extra_tag/$seqname-skel

# argv[1] gpu id
# argv[2] results dir
```
results will be saved under `ama_eval_swing/`.

### AMA-samba
```
bash scripts/eval/run_eval_ama_samba.sh 0 logdir/$extra_tag/$seqname-skel

# argv[1] gpu id
# argv[2] results dir
```
results will be saved under `ama_eval_samba/`.

### iiwa
```
bash scripts/eval/run_eval_iiwa.sh 0 logdir/$extra_tag/$seqname-skel

# argv[1] gpu id
# argv[2] results dir
```
results will be saved under `iiwa_eval/`.

### eagle
```
bash scripts/eval/run_eval_eagle.sh 0 logdir/$extra_tag/$seqname-skel

# argv[1] gpu id
# argv[2] results dir
```
results will be saved under `eagle_eval/`.