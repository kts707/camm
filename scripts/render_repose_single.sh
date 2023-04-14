# save re-posing single frame example
dev=$1
testdir=$2
kinematic_chain=$3

CUDA_VISIBLE_DEVICES=$dev python scripts/repose/repose_single.py --flagfile=$testdir/opts.log \
  --model_path $testdir/params_latest.pth \
  --skeleton_file=$kinematic_chain \
  --canonical_mesh_path=$testdir/mesh-rest.obj \
  --output_dir=$testdir/output_single