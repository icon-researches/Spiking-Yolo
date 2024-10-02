import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron

from snntorch import spikegen
from .conv import Conv
from ultralytics.nn.modules.calculator import conv_syops_counter_hook, bn_syops_counter_hook, Leaky_syops_counter_hook
from ultralytics.nn.modules.neuron import AdaptiveIFNode, AdaptiveLIFNode

__all__ = ('Spike_Conv', 'Spike_C2f', 'Spike_Bottleneck', 'Spike_SPPF')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
  """Pad to 'same' shape outputs."""
  if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

class Spike_Conv(nn.Module):
  """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

  def __init__(self, c1, c2, k=1, s=1, data={}, node='IF', ts=1, calculation=False, encode=True, p=None, g=1, d=1, act=True):
    """Initialize Conv layer with given arguments including activation."""
    super().__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
    self.bn = nn.BatchNorm2d(c2)

    allowed_keys = ['node', 'ts', 'calculation', 'encode', 'shift', 'thresh', 'mem']

    for key in data:
      if key not in allowed_keys:
        raise ValueError(f"Disallowed key '{key}' exists in data (spike_conv)")

    self.calculation = data['calculation'] if 'calculation' in data else calculation
    self.Encode = data['encode'] if 'encode' in data else encode
    self.node = data['node'] if 'node' in data else node
    self.timestep = data['ts'] if 'ts' in data else ts

    # Spiking 뉴런 추가
    if self.node == 'IF' or self.node=='if':
        self.spike_layer = neuron.IFNode()
    elif self.node == 'LIF' or self.node=='lif':
        self.spike_layer = neuron.LIFNode()
    elif self.node == 'Ad_IF' or self.node=='ad_if':
        self.spike_layer = AdaptiveIFNode()
    elif self.node == 'Ad_LIF' or self.node=='ad_lif':
        self.spike_layer = AdaptiveLIFNode()
    elif self.node == 'Soft_IF' or self.node=='soft_if':
        self.spike_layer = neuron.IFNode()
        self.thresh = data['thresh'] if 'thresh' in data else 0
        self.shift = data['shift'] if 'shift' in data else 0
        self.mem = None
    else:
        raise ValueError("Non defined neuron")
    print('th: ', self.thresh, 'sh: ', self.shift)

  def forward(self, x):
    #print("SNN Forward")
    # generate spikes from input data (x)
    if self.Encode == True:
        spikes = spikegen.rate(x, num_steps=self.timestep)
    elif self.Encode == False or self.Encode == None:
        spk_rec = []
        y = self.conv(x)
        y2 = self.bn(y)
        shape = y2.size()

        for t in range(self.timestep):
          spk = self.spike_layer(y2.flatten(1))
          # IF or LIF 계층 연산 횟수 측정
          spk_rec.append(spk)

        self.spike_layer.reset()
        # self.node.neuronal_reset(spk)

        spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)

        return spk_output
    else:
        raise ValueError("Not defined encoder value")

    spk_rec = []  # record output spikes

    # input spikes during self.timestep
    for t in range(self.timestep):
      cur_conv = self.conv(spikes[t])
      cur_bn = self.bn(cur_conv)
      spk_bn = self.spike_layer(cur_bn.flatten(1))

      spk_rec.append(spk_bn)  # record spikes

    shape = cur_bn.size()
    self.spike_layer.reset()

    spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)

    return spk_output

  def forward_fuse(self, x):
    """Perform transposed convolution of 2D data."""
    if self.Encode == True:
      if not torch.isnan(x).any():
        spikes = spikegen.rate(x, num_steps=self.timestep)
      else:
        spikes = x
        y = self.conv(x)
        act = nn.SiLU()
        #print('Original result')
        return act(y)
    elif self.Encode == False or self.Encode == None:
      spikes = x
    else:
      raise ValueError("Not defined encoder value")
    #spikes = x
    spk_rec = []  # record output spikes
    spike_sum = 0
    #print("IF")
    if self.node == 'Soft_IF' or self.node=='soft_if':
      #print("Soft IF")
      for t in range(self.timestep):
        cur_conv = self.conv(spikes[t]) # + self.shift
        #print(cur_conv.size())
        if self.mem is None:
          self.mem = torch.zeros(cur_conv.size(0), cur_conv.size(1), cur_conv.size(2), cur_conv.size(3), device=cur_conv.device)
          #print(self.mem.size())
        self.mem += cur_conv
        spike = self.mem.ge(self.thresh).float() * self.thresh
        self.mem -= spike
        spike_sum += spike
      #spk_rec.append(spike)  # record spikes
      spk_rec.append(spike_sum/self.timestep)
      shape = cur_conv.size()
      #print("shape", shape)
      spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)
      self.mem = None
      return spk_output

    # input spikes during self.timestep
    for t in range(self.timestep):
      cur_conv = self.conv(spikes[t])
      #cur_bn = cur_conv
      spk = self.spike_layer(cur_conv.flatten(1))
      spk_rec.append(spk)  # record spikes
    self.spike_layer.reset()

    shape = cur_conv.size()

    spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)

    #print('Spike Result')
    return spk_output

class Spike_Bottleneck(nn.Module):
  """Standard bottleneck."""

  def __init__(self, c1, c2, data ,shortcut=True, g=1, k=(3, 3), e=0.5):
    """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
    expansion.
    """
    super().__init__()
    c_ = int(c2 * e)  # hidden channels

    allowed_keys = ['conv1', 'conv2']

    for key in data:
      if key not in allowed_keys:
        raise ValueError(f"Disallowed key '{key}' exists in data (spike_bottleneck)")

    #{'conv1':{'node': 'if', 'ts':1}, 'conv2':{'node': 'if', 'ts':1}}
    if 'conv1' in data:
      if data['conv1'] == 'Conv' or data['conv1'] == 'conv':
        self.cv1 = Conv(c1, c_, k[0], 1)
      else:
        self.conv1 = data['conv1']
        self.cv1 = Spike_Conv(c1, c_, k[0], 1, self.conv1)

    if 'conv2' in data:
      if data['conv2'] == 'Conv' or data['conv2'] == 'conv':
        self.cv2 = Conv(c1, c_, k[0], 1)
      else:
        self.conv2 = data['conv2']
        self.cv2 = Spike_Conv(c1, c_, k[0], 1, self.conv2)

    self.add = shortcut and c1 == c2

  def forward(self, x):
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Spike_C2f(nn.Module):
  """Faster Implementation of CSP Bottleneck with 2 convolutions."""
  def __init__(self, c1, c2, data, n=1, shortcut=False, g=1, e=0.5):
    """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
    expansion.
    """
    super().__init__()
    self.c = int(c2 * e)  # hidden channels

    allowed_keys = ['first_conv', 'bn', 'last_conv']

    for key in data:
      if key not in allowed_keys:
        raise ValueError(f"Disallowed key '{key}' exists in data (spike_c2f)")

    if 'first_conv' in data:
      if data['first_conv'] == 'Conv' or data['first_conv'] == 'conv':
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
      else:
        self.first_conv = data['first_conv']
        self.cv1 = Spike_Conv(c1, 2 * self.c, 1, 1, data=self.first_conv)
    else:
      self.cv1 = Conv(c1, 2 * self.c, 1, 1)

    if 'bn' in data:
      if 'bn1' in data['bn']:
        self.bn1 = data['bn']['bn1'] #bn1:{'conv1':{'node': 'if', 'ts':1}, 'conv2':{'node': 'if', 'ts':1}}
      if 'bn2' in data['bn']:
        self.bn2 = data['bn']['bn2']
    else:
      self.bn1 = {'conv1':'conv', 'conv2':'conv'}
      self.bn2 = {'conv1': 'conv', 'conv2': 'conv'}

    # Conversion of Bottleneck's conv layers
    self.m = nn.ModuleList(
      Spike_Bottleneck(self.c, self.c, getattr(self, f'bn{j+1}'), shortcut, g, k=((3, 3), (3, 3)), e=1.0) for j in range(n))

    if 'last_conv' in data:
      if data['last_conv'] == 'Conv' or data['last_conv'] == 'conv':
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
      else:
        self.last_conv = data['last_conv']
        self.cv2 = Spike_Conv((2 + n) * self.c, c2, 1, 1, data=self.last_conv)
    else:
      print("SC2f_spike-2 : Conv")
      self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

  def forward(self, x):
    y = list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))

  def forward_split(self, x):
    """Forward pass using split() instead of chunk()."""
    y = list(self.cv1(x).split((self.c, self.c), 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))

class Spike_SPPF(nn.Module):
  """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

  def __init__(self, c1, c2, data, k=5):
    """
    Initializes the SPPF layer with given input/output channels and kernel size.

    This module is equivalent to SPP(k=(5, 9, 13)).
    """
    super().__init__()
    c_ = c1 // 2  # hidden channels

    allowed_keys = ['first_conv', 'last_conv']

    for key in data:
      if key not in allowed_keys:
        raise ValueError(f"Disallowed key '{key}' exists in data (spike_sppf)")

    if 'first_conv' in data:
      if data['first_conv'] == 'Conv' or data['first_conv'] == 'conv':
        self.cv1 = Conv(c1, c_, 1, 1)
      else:
        self.first_conv = data['first_conv']
        self.cv1 = Spike_Conv(c1, c_, 1, 1, data=self.first_conv)
    else:
      self.cv1 = Conv(c1, c_, 1, 1)

    if 'last_conv' in data:
      if data['last_conv'] == 'Conv' or data['last_conv'] == 'conv':
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
      else:
        self.last_conv = data['last_conv']
        self.cv2 = Spike_Conv(c_ * 4, c2, 1, 1, data=self.last_conv)
    else:
      self.cv2 = Conv(c_ * 4, c2, 1, 1)

    self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

  def forward(self, x):
    x = self.cv1(x)
    # self.cv1(x) 호출 결과, "Conv(c1, c_, 1, 1)" 블록의 forward 함수가 호출되면서 해당 Conv 블록의 conv 및 bn 연산 횟수가 출력된다.
    y1 = self.m(x)

    y2 = self.m(y1)

    y3 = self.m(y2)

    return self.cv2(torch.cat((x, y1, y2, y3), 1))
    # self.cv2(torch.cat((x, y1, y2, y3), 1))) 호출 결과, "Conv(c_ * 4, c2, 1, 1)" 블록의 forward 함수가 호출되면서 해당 Conv 블록의 conv 및 bn 연산 횟수가 출력된다.
