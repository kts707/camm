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
if [ "$use_human" = "no" ]; then
  pose_cnn_path=mesh_material/posenet/quad.pth
else
  pose_cnn_path=mesh_material/posenet/human.pth
fi

# mode: line load
echo line load and warmup stage
savename=${extra_tag}/${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
  $seqname $addr \
  $(cat $flagfile) \
  --warmup_shape_ep 5 --warmup_rootmlp

echo save evolution videos
python scripts/visualize/render_evolve.py --testdir logdir/$savename --first_idx 0 --last_idx $num_epochs

# mode: pose correction
# 0-80% body pose with proj loss, 80-100% gradually add all loss
# freeze shape/feature etc
echo training stage 1
loadname=${extra_tag}/${model_prefix}-init
savename=${extra_tag}/${model_prefix}-ft1
num_epochs=$((num_epochs/4))

bash scripts/template-mgpu.sh $gpus $savename \
  $seqname $addr \
  $(cat $flagfile) \
  --pose_cnn_path $pose_cnn_path --num_epochs $num_epochs \
  --model_path logdir/$loadname/params_latest.pth \
  --warmup_steps 0 --nf_reset 1 --bound_reset 1 \
  --dskin_steps 0 --fine_steps 1 --noanneal_freq \
  --freeze_proj --proj_end 1

echo save evolution videos
python scripts/visualize/render_evolve.py --testdir logdir/$savename --first_idx 0 --last_idx $num_epochs

# mode: fine tune with active+fine samples, large rgb loss wt and reset beta
echo training stage 2
loadname=${extra_tag}/${model_prefix}-ft1
savename=${extra_tag}/${model_prefix}-ft2
num_epochs=$((num_epochs*4))
bash scripts/template-mgpu.sh $gpus $savename \
  $seqname $addr \
  $(cat $flagfile) \
  --pose_cnn_path $pose_cnn_path \
  --num_epochs $num_epochs \
  --model_path logdir/$loadname/params_latest.pth \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --freeze_root --use_unc --img_wt 1 --reset_beta

echo save evolution videos
python scripts/visualize/render_evolve.py --testdir logdir/$savename --first_idx 0 --last_idx $num_epochs
