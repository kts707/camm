dev=$1
seqname=$2
modelpath=$3
vids=$4
sample_grid3d=$5
skeleton_file=$6

CUDA_VISIBLE_DEVICES=${dev} bash scripts/render_vids_skel.sh \
  ${seqname} ${modelpath} "${vids}" \
  "--sample_grid3d ${sample_grid3d} \
  --skeleton_file ${skeleton_file}
  --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"