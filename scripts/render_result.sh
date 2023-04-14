seqname=$1
model_path=$2
test_frames=$3

testdir=${model_path%/*} # %: from end
add_args=${*: 3:$#-1}
prefix=$testdir-rendering/$seqname-$test_frames

# part 1
python extract.py --flagfile=$testdir/opts.log \
                  --seqname $seqname \
                  --model_path $model_path \
                  --test_frames $test_frames \
                  $add_args
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-frz \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --freeze \
#                     --vis_cam
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-bne \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --vp -1 \
                     --vis_bones \
#                     --vis_traj
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-trj0 \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --vp 0 \
#                     --vis_traj
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-trj1 \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --vp 1 \
#                     --vis_traj
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-trj2 \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --vp 2 \
#                     --vis_traj
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-vid \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --append_img yes \
                     --append_render no
                     #--show_dp \

# part 3
ffmpeg -y -i $prefix-vid.mp4 \
          -i $prefix-frz.mp4 \
          -i $prefix-bne.mp4 \
          -i $prefix-trj0.mp4 \
          -i $prefix-trj1.mp4 \
          -i $prefix-trj2.mp4 \
-filter_complex "[0:v][1:v][2:v]hstack=inputs=3[top];\
[3:v][4:v][5:v]hstack=inputs=3[bottom];\
[top][bottom]vstack=inputs=2[v]" \
-map "[v]" \
$prefix-all.mp4

ffmpeg -y -i $prefix-all.mp4 -vf "scale=iw/2:ih/2" $prefix-all.gif
