# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .deformable_conv import DeformableConvLayer


class CCN(nn.Module):
    def __init__(self,
                 input_dims,
                 output_dims,
                 num_layers=1,
                 UAV_output_index=None,
                 Satellite_ouput_index=None,
                 use_deformable_conv=False,
                 **kwargs):
        super(CCN, self).__init__()
        self.Conv2dDict = nn.ModuleDict()
        self.num_layers = num_layers
        self.UAV_output_index = UAV_output_index if UAV_output_index is not None else []
        self.Satellite_ouput_index = Satellite_ouput_index if Satellite_ouput_index is not None else []
        self.use_deformable_conv = use_deformable_conv
        
        # 检查是否是单一索引
        if isinstance(Satellite_ouput_index, int):
            self.Satellite_ouput_index = [Satellite_ouput_index]
        else:
            self.Satellite_ouput_index = Satellite_ouput_index
            
        # 为所有特征通道转换创建卷积层
        for ind, in_channel in enumerate(input_dims):
            tmp_channels = in_channel
            Conv2dList = nn.ModuleList()
            for layer_ind in range(num_layers):      
                # 为UAV视角流添加几何畸变鲁棒性增强
                if self.use_deformable_conv and ind in self.UAV_output_index and layer_ind == num_layers - 1:
                    # 最后一层使用Deformable Conv增强几何特征提取能力
                    Conv2dList.append(DeformableConvLayer(tmp_channels, output_dims, kernel_size=3, padding=1))
                else:
                    # 其他情况使用普通卷积
                    Conv2dList.append(nn.Conv2d(tmp_channels, output_dims, kernel_size=1))
                tmp_channels = output_dims
            conv_module = nn.Sequential(*Conv2dList)
            self.Conv2dDict["neck_ccn_{}".format(ind)] = conv_module
        
        # 为卫星视角流添加多尺度融合模块
        if self.Satellite_ouput_index and len(self.Satellite_ouput_index) > 1:
            self.satellite_msf = nn.Sequential(
                nn.Conv2d(output_dims * len(self.Satellite_ouput_index), output_dims, kernel_size=1),
                nn.BatchNorm2d(output_dims),
                nn.ReLU(inplace=True)
            )
        else:
            self.satellite_msf = None
            
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output_features = []
        uav_features = []
        satellite_features = []
        
        for ind in range(len(x)):
            out_feature = x[ind]
            # 检查特征格式并确保转换为(B, C, H, W)格式
            # Swin-T输出是(B, H, W, C)，ViT-S输出应该是(B, C, H, W)
            if len(out_feature.shape) == 4:
                # 检查通道维度位置
                if out_feature.shape[1] < out_feature.shape[3]:
                    # 形状为(B, H, W, C)，转换为(B, C, H, W)
                    out_feature = out_feature.permute(0, 3, 1, 2).contiguous()
            name = "neck_ccn_{}".format(ind)
            out_feature = self.Conv2dDict[name](out_feature)
            
            if ind in self.UAV_output_index:
                uav_features.append(out_feature)
            if ind in self.Satellite_ouput_index:
                satellite_features.append(out_feature)
            if ind not in self.UAV_output_index and ind not in self.Satellite_ouput_index:
                output_features.append(out_feature)
        
        # 卫星视角流多尺度融合
        if self.satellite_msf and len(satellite_features) > 1:
            # 将所有卫星特征上采样到最大特征图大小
            max_h, max_w = satellite_features[-1].shape[2], satellite_features[-1].shape[3]
            upsampled_features = []
            
            for i, feature in enumerate(satellite_features):
                # 上采样到最大尺寸
                upsampled = F.interpolate(feature, size=(max_h, max_w), mode='bilinear', align_corners=False)
                upsampled_features.append(upsampled)
            
            # 拼接并融合
            fused_satellite = torch.cat(upsampled_features, dim=1)
            fused_satellite = self.satellite_msf(fused_satellite)
            satellite_features = [fused_satellite]
        
        # 合并所有输出特征
        output_features = uav_features + satellite_features + output_features
        return output_features
