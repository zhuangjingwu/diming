import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(DeformableConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        
        # 常规卷积层用于特征提取
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 偏移量卷积层，输出2*kh*kw个通道（x和y方向的偏移量）
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, 
                                     stride=stride, padding=padding, dilation=dilation, bias=True)
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # 初始化偏移量为0
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
    
    def forward(self, x):
        # 计算偏移量
        offset = self.offset_conv(x)
        
        # 使用torchvision.ops中的deform_conv2d函数
        output = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )
        
        return output


class DeformableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(DeformableConvLayer, self).__init__()
        
        # Deformable Conv层
        self.deform_conv = DeformableConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # 激活函数和批归一化
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.deform_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
