dev=$1
testdir=$2

## eagle
gtdir=database/DAVIS/Meshes/Full-Resolution/a-eagle-1/
gt_pmat=canonical
seqname=a-eagle
seqname_eval=a-eagle-1


# evaluation
mkdir -p eagle_eval
outfile=`cut -d/ -f2 <<<"${testdir}"`
echo results saved under eagle_eval/$seqname_eval-$outfile.txt
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval-pred \
 --seqname $seqname_eval --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat ${gt_pmat} --save_eval_plot \
 > eagle_eval/$seqname_eval-$outfile.txt
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval-gt \
 --seqname $seqname_eval --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat ${gt_pmat} --vis_gtmesh

# save to videos
ffmpeg -y -i $testdir/$seqname-eval-gt.mp4 \
          -i $testdir/$seqname-eval-pred.mp4 \
-filter_complex "[0:v][1:v]vstack=inputs=2[v]" \
-map "[v]" \
$testdir/$seqname-all.mp4