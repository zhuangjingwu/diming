import torch.optim as optim
from torch.optim import lr_scheduler
import math


def make_optimizer(model, opt):
    backbone_lr_rate = opt.lr_config.get("backbone_lr_rate", 1.0)
    if opt.model["backbone"]["type"] == "MixFormer":
        ignore_model_arch = [model.union_backbone]
    else:
        ignore_model_arch = [model.backbone_uav, model.backbone_satellite]
    ignored_params = []
    for i in ignore_model_arch:
        ignored_params += list(map(id, i.parameters()))
    extra_params = filter(lambda p: id(
        p) not in ignored_params, model.parameters())
    backbone_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    optimizer_ft = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr_rate * opt.lr_config["lr"]},
        {'params': extra_params, 'lr': opt.lr_config["lr"]}],
        weight_decay=1e-3)

    # optimizer_ft = optim.AdamW(model.parameters(),lr=opt.lr_config["lr"], weight_decay=5e-4)

    # optimizer_ft = optim.SGD(model.parameters(), lr=opt.lr , weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    # multistepsLR
    if opt.lr_config["type"]=="steps":
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=opt.lr_config["steps"], gamma=opt.lr_config["gamma"])
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=opt.num_epochs, eta_min=opt.lr*0.001)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.9)

    # ReduceLROnPlateau
    elif opt.lr_config["type"]=="ReduceLROnPlateau":
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=opt.lr_config.get("factor", 0.5), 
                                                        patience=opt.lr_config.get("patience", 4), verbose=True,
                                                        threshold=opt.lr_config.get("threshold", 0.0001), threshold_mode='rel',
                                                        cooldown=opt.lr_config.get("cooldown", 0), 
                                                        min_lr=opt.lr_config.get("min_lr", 1e-5), eps=1e-08)

    # warmupCosineLR
    elif opt.lr_config["type"]=="cosine":
        warmup_iters = opt.lr_config["warmup_iters"]
        # 估计总迭代次数 = 轮数 * 每轮迭代次数（使用batchsize和数据集大小的估计）
        # 假设数据集大小为10000，这是一个合理的估计值
        estimated_dataset_size = 10000
        iters_per_epoch = max(1, estimated_dataset_size // opt.data_config.get("batchsize", 8))
        total_iters = opt.train_config["num_epochs"] * iters_per_epoch
        final_lr_rate = opt.lr_config["warmup_ratio"]
        lambda0 = lambda iter: ((1-final_lr_rate)*iter / warmup_iters+final_lr_rate) if iter < warmup_iters else final_lr_rate \
                    if 0.5 * (1+math.cos(math.pi*(iter - warmup_iters)/(total_iters-warmup_iters))) < final_lr_rate \
                    else 0.5 * (1+math.cos(math.pi*(iter - warmup_iters)/(total_iters-warmup_iters)))
        exp_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_ft, lr_lambda=lambda0)
    else:
        raise TypeError("learning rate scheduler type of {} is not support not!!".format(opt.lr_config["type"]))

    return optimizer_ft, exp_lr_scheduler
