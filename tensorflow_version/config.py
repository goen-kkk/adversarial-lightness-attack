# import torch
# from utils import Dict
from loss import CarliniWagnerLoss, RegularizationLoss
# from xception.model_selection import TransferModel
from Meso.classifiers import *


class Dict(dict):
    """
    Example:
    m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
          for k, v in kwargs.items():
            self[k] = v

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]


cfg = Dict()

cfg.curve_steps = 64
# cfg.color_curve_range = (0.  90, 1.10)
# cfg.lab_curve_range = (-0.2, 0.8)

cfg.m = -0.1
cfg.n = 0.9
cfg.alpha = 0.3
cfg.beta = 0.3
cfg.tao = 0.2

cfg.img_size = 64
cfg.img_channels = 1
cfg.batch_size = 300
cfg.iter_n = 100

cfg.CWloss = CarliniWagnerLoss(cfg.tao)
cfg.RLloss = RegularizationLoss(cfg.curve_steps)
'''Torch'''
# cfg.CWloss_torch = CarliniWagnerLossTorch(cfg.tao)
# cfg.RLloss_torch = RegularizationLossTorch(cfg.curve_steps)
# cfg.xception = TransferModel(modelchoice='xception', num_out_classes=2)
# cfg.xception = torch.load("xception/models/x-model23.p", map_location="cuda" if torch.cuda.is_available() else "cpu")
''''''
cfg.model = Meso4()
cfg.model.load('Meso/weights/weights.h5')