# Use precomputed root body poses
gpus=$1
addr=$2
flagfile=$3
extra_tag=$4

num_epochs_configs="$(grep num_epochs $flagfile)"
num_epochs=${num_epochs_configs:13}

use_human_configs="$(grep use_human $flagfile)"
use_human=${use_human_configs:2:2}

seqname_configs="$(grep seqname $flagfile)"
seqname=${seqname_configs:10}

model_prefix=$seqname

# mode: line load
# difference from template.sh
# add use_rtk_file flag to use the poses under database/DAVIS/Cameras/
savename=${extra_tag}/${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
  $seqname $addr \
  $(cat $flagfile) \
  --use_rtk_file --freeze_root \
  --warmup_shape_ep 5 --warmup_rootmlp

echo save evolution videos
python scripts/visualize/render_evolve.py --testdir logdir/$savename --first_idx 0 --last_idx $num_epochs

# mode: pose correction
# 0-80% body pose with proj loss, 80-100% gradually add all loss
# freeze shape/feature etc
loadname=${extra_tag}/${model_prefix}-init
savename=${extra_tag}/${model_prefix}-ft1
num_epochs=$((num_epochs/4))
bash scripts/template-mgpu.sh $gpus $savename \
  $seqname $addr \
  $(cat $flagfile) \
  --num_epochs $num_epochs \
  --model_path logdir/$loadname/params_latest.pth \
  --warmup_steps 0 --nf_reset 1 --bound_reset 1 \
  --dskin_steps 0 --fine_steps 1 --noanneal_freq \
  --freeze_proj --proj_end 1

echo save evolution videos
python scripts/visualize/render_evolve.py --testdir logdir/$savename --first_idx 0 --last_idx $num_epochs

# mode: fine tune with active+fine samples, large rgb loss wt and reset beta
loadname=${extra_tag}/${model_prefix}-ft1
savename=${extra_tag}/${model_prefix}-ft2
num_epochs=$((num_epochs*4))
bash scripts/template-mgpu.sh $gpus $savename \
  $seqname $addr \
  $(cat $flagfile) \
  --num_epochs $num_epochs \
  --model_path logdir/$loadname/params_latest.pth \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --freeze_root --use_unc --img_wt 1 --reset_beta

echo save evolution videos
python scripts/visualize/render_evolve.py --testdir logdir/$savename --first_idx 0 --last_idx $num_epochs
