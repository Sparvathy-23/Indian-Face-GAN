import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.xavier_normal_(m.weight.data)
