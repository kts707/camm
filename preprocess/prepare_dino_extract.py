from absl import flags, app
import sys
sys.path.insert(0,'third_party')
sys.path.insert(0,'./')


from nnutils.train_utils import v2s_trainer

from utils.io import config_to_dataloader
from torch.utils.data import DataLoader
from nnutils.geom_utils import tensor2array
import pickle
opts = flags.FLAGS

saving_path=sys.argv[1]


def main(_):
    seqname=opts.seqname
    print('saving path',saving_path,'seqname',seqname)
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()
    impaths = data_info['impath']
    mskpaths = [i.replace('JPEGImages', 'Annotations').replace('.jpg', '.png') for i in impaths]
    data_offset = data_info['offset']

    opts_dict = {}
    opts_dict['seqname'] = opts.seqname
    opts_dict['img_size'] = opts.img_size
    opts_dict['rtk_path'] = opts.rtk_path
    opts_dict['batch_size'] = 1
    opts_dict['ngpu'] = 1
    opts_dict['preload'] = False
    opts_dict['dframe'] = [1,2,4,8,16,32]

    dataset = config_to_dataloader(opts_dict,is_eval=True)

    dataset = DataLoader(dataset,
         batch_size= 1, num_workers=0, drop_last=False, 
         pin_memory=True, shuffle=False)    
    for dat in dataset.dataset.datasets:
        dat.spec_dt = 1
    
    collect_dict = {'imgpaths':impaths, 'mskpaths':mskpaths}

    # hardcoded path 
    frameid_list, dt_list = [], []
    for i, batch in enumerate(dataset):
        print(i)
        frameid = batch['frameid']
        dataid = batch['dataid']
        dt = frameid[0,1] - frameid[0,0]
        frameid = frameid + data_offset[dataid[0,0].long()]
        frameid_list.append(frameid.long().tolist())
        dt_list.append(dt.item())

    
    collect_dict['frameid_list'] = frameid_list
    collect_dict['dt_list'] = dt_list
    print(collect_dict, len(dt_list), len(frameid_list))
    with open(saving_path, 'wb') as f:
        pickle.dump(collect_dict, f)



if __name__ == '__main__':
    app.run(main)
