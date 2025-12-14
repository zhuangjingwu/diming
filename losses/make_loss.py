import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from losses.clsloss import BalanceLoss, CenterBalanceLoss, FocalLoss, CrossEntropyLoss, GaussianFocalLoss
from losses.regloss import SmoothL1Loss
from losses.locloss import LocSmoothL1Loss
from losses.triplet_loss import Tripletloss


def make_loss(opt):
    loss = Loss(opt)
    return loss


class HaversineLoss(nn.Module):
    """
    Haversine distance loss for latitude and longitude regression
    """
    def __init__(self, reduction='mean'):
        super(HaversineLoss, self).__init__()
        self.reduction = reduction
        # Earth radius in kilometers
        self.earth_radius = 6371.0
    
    def forward(self, pred, target):
        """
        pred: predicted latitude and longitude (batch_size, 2) in degrees
        target: ground truth latitude and longitude (batch_size, 2) in degrees
        """
        # Convert degrees to radians
        pred_rad = torch.deg2rad(pred)
        target_rad = torch.deg2rad(target)
        
        # Extract latitudes and longitudes
        pred_lat, pred_lon = pred_rad[:, 0], pred_rad[:, 1]
        target_lat, target_lon = target_rad[:, 0], target_rad[:, 1]
        
        # Differences in coordinates
        dlat = target_lat - pred_lat
        dlon = target_lon - pred_lon
        
        # Haversine formula
        a = torch.sin(dlat/2)**2 + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        distance = self.earth_radius * c
        
        # Apply reduction
        if self.reduction == 'mean':
            return distance.mean()
        elif self.reduction == 'sum':
            return distance.sum()
        else:
            return distance


class HuberLoss(nn.Module):
    """
    Huber损失 - 对异常值更鲁棒
    在误差较小时使用平方损失，误差较大时使用线性损失
    """
    def __init__(self, delta=1.0, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, pred, target):
        # 计算预测值与目标值的差异
        diff = pred - target
        # 计算绝对误差
        abs_diff = torch.abs(diff)
        # 对小误差使用平方损失，大误差使用线性损失
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        # 计算损失
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cls_loss = None
        self.reg_loss = None
        self.loc_loss = None
        self.triplet_loss = None
        self.haversine_loss = None
        self.smoothl1_loss = None
        self.huber_loss = None
        # build class loss
        if "cls_loss" in opt.model["loss"]:
            self.cls_loss = self.build_cls_loss(opt.model["loss"]["cls_loss"])
        # build regression loss
        if "reg_loss" in opt.model["loss"]:
            self.reg_loss = self.build_reg_loss(opt.model["loss"]["reg_loss"])
        if "loc_loss" in opt.model["loss"]:
            self.loc_loss = self.build_loc_loss(opt.model["loss"]["loc_loss"])
        if "triplet_loss" in opt.model["loss"]:
            self.triplet_loss = self.build_triplet_loss(opt.model["loss"]["triplet_loss"])
        if "haversine_loss" in opt.model["loss"]:
            self.haversine_loss = self.build_haversine_loss(opt.model["loss"]["haversine_loss"])
        if "smoothl1_loss" in opt.model["loss"]:
            self.smoothl1_loss = self.build_smoothl1_loss(opt.model["loss"]["smoothl1_loss"])
        if "huber_loss" in opt.model["loss"]:
            self.huber_loss = self.build_huber_loss(opt.model["loss"]["huber_loss"])
        
        # 多任务权重配置
        self.weights = opt.model["loss"].get("weights", {})
        self.cls_weight = self.weights.get("cls_weight", 1.0)
        self.reg_weight = self.weights.get("reg_weight", 1.0)
        self.haversine_weight = self.weights.get("haversine_weight", 1.0)
        self.smoothl1_weight = self.weights.get("smoothl1_weight", 1.0)
        self.triplet_weight = self.weights.get("triplet_weight", 1.0)
        self.loc_weight = self.weights.get("loc_weight", 1.0)
        self.huber_weight = self.weights.get("huber_weight", 1.0)

    def build_cls_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "BalanceLoss":
            loss_func = BalanceLoss(**loss_opt)
        elif loss_type == "CenterBalanceLoss":
            loss_func = CenterBalanceLoss(**loss_opt)
        elif loss_type == "FocalLoss":
            loss_func = FocalLoss(**loss_opt)
        elif loss_type == "CrossEntropyLoss":
            loss_func = CrossEntropyLoss(**loss_opt)
        elif loss_type == "GaussianFocalLoss":
            loss_func = GaussianFocalLoss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func

    def build_reg_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "SmoothL1Loss":
            loss_func = SmoothL1Loss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func
    
    def build_loc_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "LocSmoothL1Loss":
            loss_func = LocSmoothL1Loss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func
        
    def build_triplet_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "TripletLoss":
            loss_func = Tripletloss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func
        
    def build_haversine_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "HaversineLoss":
            loss_func = HaversineLoss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func
        
    def build_smoothl1_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "SmoothL1Loss":
            loss_func = nn.SmoothL1Loss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func
        
    def build_huber_loss(self, loss_opt):
        loss_type = loss_opt.pop("type")
        if loss_type == "HuberLoss":
            loss_func = HuberLoss(**loss_opt)
        else:
            raise NameError("{} not in the loss list!!!".format(loss_type))
        return loss_func

    def forward(self, input, target):
        """
        动态多任务损失计算
        input: (coarse_cls, fine_reg)
        target: (center_rate, gt_latlon)
        """
        coarse_cls, fine_reg = input
        ratex, ratey = target  # target包含[ratex, ratey]两个张量
        center_rate = [ratex, ratey]  # 保持与原有代码兼容
        
        total_loss = 0.0
        loss_dict = {}
        
        # 粗匹配损失
        if self.cls_loss is not None and coarse_cls is not None:
            cls_loss = self.cls_loss(coarse_cls, center_rate)
            total_loss += self.cls_weight * cls_loss
            loss_dict["cls_loss"] = cls_loss.item()
        
        # 定位损失 - 使用LocSmoothL1Loss计算实际位置偏移
        if self.loc_loss is not None and coarse_cls is not None:
            loc_loss = self.loc_loss(coarse_cls, center_rate)
            total_loss += self.loc_weight * loc_loss
            loss_dict["loc_loss"] = loc_loss.item()
        
        # 精匹配损失 - 结合Smooth L1 Loss和Huber Loss提高鲁棒性
        if self.smoothl1_loss is not None and fine_reg is not None:
            # 计算fine_reg与目标的损失
            smoothl1_loss = self.smoothl1_loss(fine_reg, torch.zeros_like(fine_reg))
            # 确保smoothl1_loss不为0
            if smoothl1_loss.item() < 1e-8:
                # 如果损失太小，添加一个小的正则化项
                smoothl1_loss = torch.mean(torch.square(fine_reg)) * 0.1
            total_loss += self.smoothl1_weight * smoothl1_loss
            loss_dict["smoothl1_loss"] = smoothl1_loss.item()
        
        # Huber Loss - 增加鲁棒性，减少异常值影响
        if self.huber_loss is not None and fine_reg is not None:
            # 使用Huber Loss计算精匹配损失，对异常值更鲁棒
            huber_loss = self.huber_loss(fine_reg, torch.zeros_like(fine_reg))
            total_loss += self.huber_weight * huber_loss
            loss_dict["huber_loss"] = huber_loss.item()
        
        # 三元组损失（可选）
        if self.triplet_loss is not None:
            triplet_loss = self.triplet_loss()
            total_loss += self.triplet_weight * triplet_loss
            loss_dict["triplet_loss"] = triplet_loss.item()
        
        return total_loss, loss_dict