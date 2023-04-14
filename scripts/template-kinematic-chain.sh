gpus=$1
addr=$2
flagfile=$3
extra_tag=$4
skeleton_file=$5

seqname_configs="$(grep seqname $flagfile)"
seqname=${seqname_configs:10}

model_prefix=$seqname

echo kinematic chain aware optimization stage
loadname=${extra_tag}/${model_prefix}-ft2
savename=${extra_tag}/${model_prefix}-skel

bash scripts/template-mgpu.sh $gpus $savename \
  $seqname $addr \
  $(cat $flagfile) \
  --skeleton_file $skeleton_file \
  --model_path logdir/$loadname/params_latest.pth \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --freeze_root --use_unc --img_wt 1

echo save evolution videos
python scripts/visualize/render_evolve.py --testdir logdir/$savename --first_idx 0 --last_idx 120 $num_epochs --mesh_only

python scripts/visualize/render_evolve.py --testdir logdir/$savename --first_idx 0 --last_idx 120 $num_epochs --draw_skel --bone_only