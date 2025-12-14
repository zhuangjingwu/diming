# 1. 模型配置(models) =========================================
model = dict(
    backbone=dict(type="Swin-T", pretrain=True, output_index=[0, 1, 2, 3], patch_size=4, window_size=7),  # 输出4个尺度的特征
    neck=dict(
        type="CCN",
        output_dims=128,
        UAV_output_index=[0],  # 使用最后一个尺度作为UAV特征
        Satellite_ouput_index=[0, 1, 2, 3],  # 卫星使用所有尺度进行多尺度融合
        use_deformable_conv=True,  # 是否使用Deformable Conv增强UAV几何畸变鲁棒性
    ),
    head=dict(
        type="AttentionFusionLib",
        fusion_type="CAT_SA",
        head_pool="linear",
        dropout_rate=0.5,
        input_ndim=128,
        mid_ndim=256,  # 中间层的维度
        attention_layer_num=4,
        linear_layer_num=1,
        enable_position_embedding=True,
        pos_length=[128, 384],
        patch_size=4,
    ),
    postprocess=dict(
        upsample_to_original=False,
    ),
    loss=dict(
        cls_loss=dict(type="GaussianFocalLoss", neg_weight=15, radius=3),
        loc_loss=dict(type="LocSmoothL1Loss", topk=1, weight_rate=1.0),
        smoothl1_loss=dict(type="SmoothL1Loss", reduction="mean"),
        haversine_loss=dict(type="HaversineLoss", reduction="mean"),
        huber_loss=dict(type="HuberLoss", delta=1.0, reduction="mean"),
        weights=dict(
            cls_weight=1.0,
            loc_weight=0.5,
            smoothl1_weight=0.3,
            haversine_weight=0.5,
            huber_weight=0.2,
        ),
    ),
)


# 2. 数据集配置(datasets) =========================================
data_config = dict(
    batchsize=8,
    num_worker=8,
    val_batchsize=8,
    train_dir="/root/autodl-tmp/University-Release/train",
    # val_dir='/root/autodl-tmp/University-Release/val',
    val_dir="/root/autodl-tmp/University-Release/test",
    test_dir="/root/autodl-tmp/University-Release",
    test_mode="test",
    UAVhw=[128, 128],
    Satellitehw=[384, 384],
)

pipline_config = dict(
    train_pipeline=dict(
        UAV=dict(
            RotateAndCrop=dict(rate=0.7),
            HorizontalFlip=dict(probability=0.5),
            VerticalFlip=dict(probability=0.5),
            RandomRotate90=dict(),
            RandomAffine=dict(degrees=360, scale=(0.7, 1.3), shear=20),
            ColorJitter=dict(brightness=0.6, contrast=0.4, saturation=0.4, hue=0.2),
            RandomErasing=dict(probability=0.7, sl=0.02, sh=0.3, r1=0.3),
            RandomResize=dict(img_size=data_config["UAVhw"]),
            ToTensor=dict(),
        ),
        Satellite=dict(
            HorizontalFlip=dict(probability=0.5),
            VerticalFlip=dict(probability=0.5),
            RandomRotate90=dict(),
            RandomAffine=dict(degrees=30, scale=(0.85, 1.15)),
            ColorJitter=dict(brightness=0.6, contrast=0.4, saturation=0.4, hue=0.2),
            RandomErasing=dict(probability=0.7, sl=0.02, sh=0.3, r1=0.3),
            RandomCrop=dict(cover_rate=0.7, map_size=(512, 1000)),
            RandomResize=dict(img_size=data_config["Satellitehw"]),
            ToTensor=dict(),
        ),
    ),
)

# 3. 训练策略配置(schedules) =========================================
lr_config = dict(lr=5e-5, type="ReduceLROnPlateau", factor=0.5, patience=4, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5)
train_config = dict(autocast=True, num_epochs=25)
test_config = dict(
    num_worker=8,
    filterR=1,
    checkpoint="output/net_best.pth",
)

# 4. 运行配置(runtime) =========================================
checkpoint_config = dict(
    interval=1,
    epoch_start_save=1,
    only_save_best=True,
)
log_interval = 50
load_from = None
resume_from = None
debug = True
seed = 666
