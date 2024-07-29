import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron

from snntorch import spikegen

from ultralytics.nn.modules.calculator import conv_syops_counter_hook, bn_syops_counter_hook, Leaky_syops_counter_hook
from ultralytics.nn.modules.neuron import AdaptiveIFNode, AdaptiveLIFNode

__all__ = ('Spike_conv', 'Spike_C2F')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
  """Pad to 'same' shape outputs."""
  if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

class Spike_conv(nn.Module):
  """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

  def __init__(self, c1, c2, k=1, s=1, data={}, node='IF', ts=1, calculation=False, encode=True, p=None, g=1, d=1, act=True):
    """Initialize Conv layer with given arguments including activation."""
    super().__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
    self.bn = nn.BatchNorm2d(c2)

    allowed_keys = ['node', 'ts', 'calculation', 'encode']

    for key in data:
      if key not in allowed_keys:
        raise ValueError(f"Disallowed key '{key}' exists in data (spike_conv)")

    self.calculation = data['calculation'] if 'calculation' in data else calculation
    self.Encode = data['encode'] if 'encode' in data else encode
    self.node = data['node'] if 'node' in data else node
    self.timestep = data['ts'] if 'ts' in data else ts

    # Spiking 뉴런 추가
    if self.node == 'IF':
        self.spike_layer = neuron.IFNode()
    elif self.node == 'LIF':
        self.spike_layer = neuron.LIFNode()
    elif self.node == 'Ad_IF':
        self.spike_layer = AdaptiveIFNode()
    elif self.node == 'Ad_LIF':
        self.spike_layer = AdaptiveLIFNode()
    else:
        raise ValueError("Non defined neuron")


  def forward(self, x):
    # generate spikes from input data (x)
    if self.Encode == True:
        spikes = spikegen.rate(x, num_steps=self.timestep)
    elif self.Encode == False or self.Encode == None:
        spikes = x
    else:
        raise ValueError("Not defined encoder value")

    spk_rec = []  # record output spikes

    # input spikes during self.timestep
    for t in range(self.timestep):
      cur_conv = self.conv(spikes[t])
      cur_bn = self.bn(cur_conv)
      spk_bn = self.spike_layer(cur_bn.flatten(1))

      if self.calculation == True:
        # conv 계층 연산 횟수 측정
        conv_syops = conv_syops_counter_hook(self.conv, spikes[t], cur_conv, "sconv_conv")
        # lif_conv(Leaky) 계층 연산 횟수 측정
        # lif_conv_syops = Leaky_syops_counter_hook(self.lif_conv, cur_conv, "sconv_lif_conv")
        # bn 계층 연산 횟수 측정
        bn_syops = bn_syops_counter_hook(self.bn, cur_conv, cur_bn, "sconv_bn")
        # lif_bn(Leaky) 계층 연산 횟수 측정
        lif_bn_syops = Leaky_syops_counter_hook(self.lif_bn, cur_bn, "sconv_lif_bn")

      spk_rec.append(spk_bn)  # record spikes

    shape = cur_bn.size()
    self.spike_layer.reset()

    spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)

    return spk_output

class Spike_Bottleneck(nn.Module):
  """Standard bottleneck."""

  def __init__(self, c1, c2, data ,shortcut=True, calculation=False ,g=1, k=(3, 3), e=0.5):
    """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
    expansion.
    """
    super().__init__()
    c_ = int(c2 * e)  # hidden channels

    # Conversion of first conv layer
    if 0 in spk_conv_li:
      print("SBottleneck_spike-1 : SConv_AT")
      self.cv1 = SConv_AT(c1, c_, k[0], 1, calculation)
    else:
      print("SBottleneck_spike-1 : Conv")
      self.cv1 = Conv(c1, c_, k[0], 1, calculation)

    # Conversion of last conv layer
    if 1 in spk_conv_li:
      print("SBottleneck_spike-2 : SConv_AT")
      self.cv2 = SConv_AT(c_, c2, k[1], 1, calculation, g=g)
    else:
      print("SBottleneck_spike-2 : Conv")
      self.cv2 = Conv(c_, c2, k[1], 1, calculation, g=g)

    self.add = shortcut and c1 == c2
    self.calculation = calculation

  def forward(self, x):
    if self.calculation == True:
      print("#=====SBottleneck_spike Block=====#")
    """'forward()' applies the YOLO FPN to input data."""
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Spike_C2F(nn.Module):
  """Faster Implementation of CSP Bottleneck with 2 convolutions."""
  def __init__(self, c1, c2, data, n=1, shortcut=False, calculation=False ,g=1, e=0.5):
    """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
    expansion.
    """
    super().__init__()
    self.c = int(c2 * e)  # hidden channels
    #data example: {'first_conv':{'node': 'if', 'ts':1}, 'bb1':{'conv1':{'node': 'if', 'ts':1}}, 'last_conv':{'node': 'if', 'ts':1}}
    # Conversion of first conv layer

    allowed_keys = ['first_conv', 'bb1', 'bb2', 'last_conv']

    for key in data:
      if key not in allowed_keys:
        raise ValueError(f"Disallowed key '{key}' exists in data (spike_c2f)")

    if 'first_conv' in data:
      if data['first_conv'] == 'Conv' or data['first_conv'] == 'conv':
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, calculation)
      else:
        self.first_conv = data['first_conv']
        self.cv1 = Spike_conv(c1, 2 * self.c, 1, 1, data=self.first_conv)
    else:
      self.cv1 = Conv(c1, 2 * self.c, 1, 1, calculation)

    if 'bb1' in data:
      self.bb1 = data['bb1']

    if 'last_conv' in data:
      if data['last_conv'] == 'Conv' or data['last_conv'] == 'conv':
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1, calculation)
      else:
        self.last_conv = data['last_conv']
        self.cv2 = Spike_conv((2 + n) * self.c, c2, 1, 1, data=self.last_conv)
    else:
      print("SC2f_spike-2 : Conv")
      self.cv2 = Conv((2 + n) * self.c, c2, 1, 1, calculation)

    # Conversion of Bottleneck's conv layers
    self.m = nn.ModuleList(
      SBottleneck_AT(self.c, self.c, spk_conv_li[j+1] ,shortcut, calculation ,g, k=((3, 3), (3, 3)), e=1.0) for j in range(n))

    self.calculation = calculation

  def forward(self, x):
    y = list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))

  def forward_split(self, x):
    """Forward pass using split() instead of chunk()."""
    y = list(self.cv1(x).split((self.c, self.c), 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))