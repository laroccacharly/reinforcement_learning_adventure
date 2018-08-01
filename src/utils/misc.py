from collections import namedtuple
import numpy as np
import torch
from torch.autograd import Variable

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'next_action', 'done'))

def to_var(x):
    """
        Because PyTorch works with Variables
    """
    x = np.array(x)
    x = torch.Tensor(x)
    if torch.cuda.is_available():
        x.cuda()
    x = Variable(x)
    return x