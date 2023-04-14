# save re-posing sequence
dev=$1
testdir=$2
kinematic_chain=$3

CUDA_VISIBLE_DEVICES=$dev python scripts/repose/repose_sequence.py --flagfile=$testdir/opts.log \
  --model_path $testdir/params_latest.pth \
  --skeleton_file=$kinematic_chain \
  --canonical_mesh_path=$testdir/mesh-rest.obj \
  --output_dir=$testdir/output_sequence