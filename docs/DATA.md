## Data Preparation

### Data format
We follow the same format as BANMo. We also save DINO features as lines of pixels to speed up dataloading.

```
DAVIS/
    JPEGImages/
        Full-Resolution/
            sequence-name/
                {%05d}.jpg
    # segmentations from detectron2
    Annotations/
        Full-Resolution/
            sequence-name/
                {%05d}.png
    # forward backward flow between every {1,2,4,8,16,32} frames from VCN-robust
    FlowBW_%d/ and FlowFw_%d/ 
        Full-Resolution/
            sequence-name/ and optionally seqname-name_{%02d}/ (frame interval)
                flo-{%05d}.pfm
                occ-{%05d}.pfm
                visflo-{%05d}.jpg
                warp-{%05d}.jpg
    # 16-dim Densepose features from CSE
    Densepose/
        Full-Resolution/
            sequence-name/
                # 112x(112*16) cropped densepose features
                feat-{%05d}.pfm 
                # [x,y,w,h] saved to warp cropped features to original coordinate
                bbox-{%05d}.txt 
                # densepose surface indices, for visualization
                {%05d}.pfm 
    # lines of pixels in order to speed up dataloading
    Pixels/  
        Full-Resolution/
            sequence-name/
                # skipped frames of flow followed by frame index
                %d-%05d/ 
    # lines of pixels for DINO features
    DINO_Pixels/  
        Full-Resolution/
            sequence-name/
                # skipped frames of flow followed by frame index
                %d-%05d/ 
```


### Download optical flow model and PoseNet weights from BANMo
```
# optical flow model
mkdir ./lasr_vcn
wget https://www.dropbox.com/s/bgsodsnnbxdoza3/vcn_rob.pth -O ./lasr_vcn/vcn_rob.pth

# PoseNet weights
mkdir -p mesh_material/posenet && cd "$_"
wget $(cat ../../misc/posenet.txt); cd ../../
```

### Download RigNet weights and set up RigNet folder
Download the RigNet weights from [here](https://umass-my.sharepoint.com/personal/zhanxu_umass_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhanxu%5Fumass%5Fedu%2FDocuments%2FBoxData%2Ftrained%5Fmodels%2Ezip&parent=%2Fpersonal%2Fzhanxu%5Fumass%5Fedu%2FDocuments%2FBoxData&ga=1) and unzip it under third_party/RigNet. Also create a folder to store the meshes as third_party/input_meshes
```
third_party/
    RigNet/
        checkpoints/
        input_meshes/
        other folders and files...
```

### iiwa Dataset
Download our iiwa dataset from [here](https://camm3d.github.io/) and place it at raw/iiwa.zip
```
# unzip the dataset
cd raw
unzip iiwa.zip
cd ..

# preprocess the dataset
bash preprocess/preprocess_iiwa.sh iiwa .mp4 24

# store as lines
seqname=iiwa
python preprocess/img2lines.py --seqname $seqname

# DINO features
python preprocess/prepare_dino_extract.py data_info_iiwa.pkl --seqname $seqname
python preprocess/compute_dino.py data_info_iiwa.pkl 16

```


### AMA-Human Dataset
Download swing and samba sequences from [AMA dataset website](https://people.csail.mit.edu/drdaniel/mesh_animation/) or 
run the following scripts:
```
cd database; wget $(cat ../misc/ama.txt);
# untar files
ls *.tar | xargs -i tar xf {}
find ./T_* -type f -name "*.tgz" -execdir tar -xvzf {} \;
cd ../
```
Convert the data into correct format:
```
python scripts/ama-process/ama2davis.py --path ./database/T_samba
python scripts/ama-process/ama2davis.py --path ./database/T_swing
```
Then extract flow and dense appearance features:
```
seqname=ama-female
mkdir raw/$seqname;
# write filenames in replace of .MOV files
ls -d database/DAVIS/Annotations/Full-Resolution/T_s* | xargs -i echo {} | sed 's:.*/::' | xargs -i touch raw/$seqname/{}.txt # create empty txt files
bash preprocess/preprocess.sh $seqname .txt y 10

# store as lines
python preprocess/img2lines.py --seqname $seqname 
```
To use DINO features, extract and save DINO features by running the following scripts:
```
python preprocess/prepare_dino_extract.py data_info_ama.pkl --seqname $seqname
python preprocess/compute_dino.py data_info_ama.pkl 16
```

### Eagle Dataset
First install soft rasterizer
```
pip install -e third_party/softras
```

Then download animated mesh sequences
```
mkdir database/eagle && cd "$_"
wget https://www.dropbox.com/sh/xz8kckfq817ggqd/AADIhtb1syWhDQeY8xa9Brc0a -O eagle.zip
unzip eagle.zip; cd ../../
```

Render image data and prepare mesh ground-truth
```
bash scripts/synthetic/render_eagle.sh
``` 

Store as lines and save DINO features
```
seqname=a-eagle
# store as lines
python preprocess/img2lines.py --seqname $seqname

# DINO features
python preprocess/prepare_dino_extract.py data_info_eagle.pkl --seqname $seqname
python preprocess/compute_dino.py data_info_eagle.pkl 16
```

### Custom Dataset
Place all the RGB videos under raw/my_data
```
# preprocess the dataset
bash preprocess/preprocess.sh my_data .mp4 24

# store as lines
seqname=my_data
python preprocess/img2lines.py --seqname $seqname

# DINO features
python preprocess/prepare_dino_extract.py data_info_iiwa.pkl --seqname $seqname
python preprocess/compute_dino.py data_info_my_data.pkl 16

```
