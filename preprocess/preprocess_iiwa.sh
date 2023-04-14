#
# bash preprocess/preprocess_iiwa.sh iiwa .mp4 24 

rootdir=raw
tmpdir=tmp/
prefix=$1
filedir=$rootdir/$prefix
maskoutdir=$rootdir/output
finaloutdir=database/DAVIS/
suffix=$2
fps=$3

# create required dirs
mkdir ./tmp
mkdir -p database/DAVIS/
mkdir -p raw/output

counter=0
for infile in `ls -v $filedir/videos/*$suffix`; do
  echo $infile

  if [ "$suffix" = ".MOV" ] || [ "$suffix" = ".mp4" ]; then
    seqname=$prefix$(printf "%03d" $counter)
    ## process videos
    # extract frames
    rm -rf $maskoutdir
    mkdir -p $maskoutdir
    ffmpeg -i $infile -vf fps=$fps $maskoutdir/%05d.jpg

    # segmentation
    todir=$tmpdir/$seqname
    rm -rf $todir
    mkdir $todir
    mkdir $todir/images/
    mkdir $todir/masks/

    echo $todir/images
    cp $maskoutdir/* $todir/images
    rm -rf $finaloutdir/JPEGImages/Full-Resolution/$seqname  
    rm -rf $finaloutdir/Annotations/Full-Resolution/$seqname 
    rm -rf $finaloutdir/Densepose/Full-Resolution/$seqname   
    mkdir -p $finaloutdir/JPEGImages/Full-Resolution/$seqname
    mkdir -p $finaloutdir/Annotations/Full-Resolution/$seqname
    mkdir -p $finaloutdir/Densepose/Full-Resolution/$seqname

  fi
  counter=$((counter+1))
done

counter=0
for infile in `ls -v $filedir/masks/*$suffix`; do
  echo $infile  

  if [ "$suffix" = ".MOV" ] || [ "$suffix" = ".mp4" ]; then
    seqname=$prefix$(printf "%03d" $counter)
    ## process videos
    # extract frames
    todir=$tmpdir/$seqname
    ffmpeg -i $infile -vf fps=$fps $maskoutdir/%05d.jpg
    echo $todir/masks
    cp $maskoutdir/* $todir/masks

    python preprocess/mask_iiwa.py $seqname
  fi

  python preprocess/compute_dp.py $seqname y

  # flow
  echo $seqname
  cd third_party/vcnplus
  bash compute_flow.sh $seqname
  cd -

  counter=$((counter+1))
done

# write config file
# python preprocess/write_config.py iiwa no