"""
@author: Shikuang Deng
"""
import torch
import torch.nn as nn

class CNNtoSNN(nn.Module):
    def __init__(self, thresh, Conv2d):
        super(SPIKE_layer, self).__init__()
        self.thresh = thresh
        self.ops = Conv2d
        self.mem = 0
        if args.shift_snn > 0:
            self.shift = self.thresh / (2 * args.shift_snn)
        else:
            self.shift = 0
        # self.shift_operation()

    # Initialize the membrane potential
    def init_mem(self):
        self.mem = 0

    def shift_operation(self):
        for key in self.state_dict().keys():
            if 'bias' in key:
                pa = self.state_dict()[key]
                pa.copy_(pa + self.shift)

    def forward(self, input):
        # spiking
        x = self.ops(input) + self.shift
        self.mem += x
        spike = self.mem.ge(self.thresh).float() * self.thresh
        # soft-rest
        self.mem -= spike
        return spike

