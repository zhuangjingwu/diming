import torch.nn as nn
import numpy as np
from .Backbone.backbone import make_backbone
from .Neck.neck import make_neck
from .Head.head import make_head
from .PostProcess.postprocess import make_postprocess
import time
import torch
from torch.nn import functional as F



class FPI(nn.Module):
    def __init__(self, opt):
        super(FPI, self).__init__()
        self.opt = opt
        # backbone init
        self.backbone_name = opt.model["backbone"]["type"]
        if self.backbone_name == "MixFormer":
            opt.model["backbone"]["satellite_size"] = opt.data_config["Satellitehw"][0]
            opt.model["backbone"]["uav_size"] = opt.data_config["UAVhw"][0]
            self.union_backbone = make_backbone(opt)
            opt.backbone_output_channel = self.union_backbone.backbone_out_channel
        else:
            self.backbone_uav = make_backbone(opt, opt.data_config["UAVhw"])
            share = opt.model["backbone"].get("share", False)
            if share:
                self.backbone_satellite = self.backbone_uav
            else:
                self.backbone_satellite = make_backbone(opt, opt.data_config["Satellitehw"])
            opt.backbone_output_channel = self.backbone_uav.backbone_out_channel

        # neck init
        self.neck_uav = make_neck(opt)
        self.neck_satellite = make_neck(opt)
        self.UAV_output_index = opt.model["neck"]["UAV_output_index"]
        
        # 确保Satellite_output_index是一个列表
        satellite_index = opt.model["neck"]["Satellite_ouput_index"]
        self.Satellite_output_index = [satellite_index] if isinstance(satellite_index, int) else satellite_index

        # head init
        opt.model["head"]["muti_level_nums"] = len(opt.model["neck"]["UAV_output_index"])
        self.head = make_head(opt)

        # upsample init
        self.upsample_to_original = opt.model["postprocess"]["upsample_to_original"]
        if self.upsample_to_original:
            self.postprocess = make_postprocess(opt)

    def forward(self, z, x):
        # backbone forward
        if self.backbone_name == "MixFormer":
            z, x = self.union_backbone(x, z)
        else:
            z = self.backbone_uav(z)
            x = self.backbone_satellite(x)
        # neck forward
        neck_z = self.neck_uav(z)
        # UAV特征现在是neck_z的第一部分
        neck_z = neck_z[:len(self.UAV_output_index)]
        # 处理卫星特征
        neck_x = self.neck_satellite(x)
        # 卫星特征现在是neck_x的前部分，与UAV特征分离后的部分
        neck_x = neck_x[len(self.UAV_output_index):len(self.UAV_output_index)+len(self.Satellite_output_index)]
        # 如果是多尺度融合后的单一特征图，则取第一个
        if len(neck_x) == 1:
            neck_x = neck_x[0]
        # head forward
        cls_out, reg_out = self.head(neck_z, neck_x)
        # postprocess forward
        if self.upsample_to_original:
            cls_out,reg_out = self.postprocess(cls_out,reg_out)

        return cls_out, reg_out

    def load_checkpoint(self, checkpoint_path=""):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)
        print("Load pretrained backbone checkpoint from:", checkpoint_path)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)


def make_model(opt):
    # init the FPI model
    model = FPI(opt)
    # if 'load_from' is not empty, load the pretrain checkpoint.
    if isinstance(opt.load_from, str) and len(opt.load_from) > 0:
        model.load_checkpoint(opt.load_from)
    return model
