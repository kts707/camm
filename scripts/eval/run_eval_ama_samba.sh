dev=$1
testdir=$2

## T_samba
gtdir=database/T_samba/meshes/
gt_pmat=database/T_samba/calibration/Camera1.Pmat.cal
seqname=ama-female
seqname_eval=T_samba1

# evaluation
mkdir -p ama_eval_samba
outfile=`cut -d/ -f2 <<<"${testdir}"`
echo results saved under ama_eval_samba/$seqname_eval-$outfile.txt
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval-pred \
 --seqname $seqname_eval --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat ${gt_pmat} --save_eval_plot \
 > ama_eval_samba/$seqname_eval-$outfile.txt
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval-gt \
 --seqname $seqname_eval --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat ${gt_pmat} --vis_gtmesh

# save to videos
ffmpeg -y -i $testdir/$seqname-eval-gt.mp4 \
          -i $testdir/$seqname-eval-pred.mp4 \
-filter_complex "[0:v][1:v]vstack=inputs=2[v]" \
-map "[v]" \
$testdir/$seqname-all.mp4