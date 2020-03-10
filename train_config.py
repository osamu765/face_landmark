

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()
#### below are params for dataiter
config.TRAIN.process_num = 5
config.TRAIN.prefetch_size = 20
############

config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 128
config.TRAIN.log_interval = 10                  ##10 iters for a log msg
config.TRAIN.epoch = 1000

config.TRAIN.lr_value_every_epoch = [0.00001,0.0001,0.001,0.0001,0.00001,0.000001,0.0000001]          ####lr policy
config.TRAIN.lr_decay_every_epoch = [1,2,100,150,200,250]
config.TRAIN.weight_decay_factor = 5.e-4                                    ####l2
config.TRAIN.vis=False                                                      #### if to check the training data
config.TRAIN.mix_precision=False                                            ##use mix precision to speedup, tf1.14 at least
config.TRAIN.opt='Adam'                                                     ##Adam or SGD

config.MODEL = edict()
config.MODEL.model_path = './model/'                                        ## save directory
config.MODEL.hin = 64                                                      # input size during training , 128,160,   depends on
config.MODEL.win = 64
config.MODEL.out_channel=2+3    # output vector    68 points , 3 headpose ,4 cls params,(left eye, right eye, mouth, big mouth open)

#### 'ShuffleNetV2_1.0' 'ShuffleNetV2_0.5' or MobileNetv2,
config.MODEL.net_structure='ShuffleNetV2_0.75'
config.MODEL.pretrained_model=None
config.DATA = edict()

config.DATA.root_path=''
config.DATA.train_txt_path='train.json'
config.DATA.val_txt_path='val.json'

############the model is trained with RGB mode
config.DATA.PIXEL_MEAN = [127., 127., 127.]             ###rgb
config.DATA.PIXEL_STD = [127., 127., 127.]              

config.DATA.base_extend_range=[0.2,0.3]                 ###extand
config.DATA.scale_factor=[0.7,1.35]                     ###scales

config.DATA.symmetry = [(0, 0)]


weights=[1.]
weights_xy=[[x,x] for x in weights]

config.DATA.weights = np.array(weights_xy,dtype=np.float32).reshape([-1])


config.MODEL.pruning=False               ## pruning flag  add l1 reg to bn/beta, no use for tmp
config.MODEL.pruning_bn_reg=0.00005



