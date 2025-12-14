# 1. 模型配置(models) =========================================
model = dict(
    backbone=dict(type="Swin-T", pretrain=True, output_index=[0, 1, 2, 3]),  # 输出4个尺度的特征
    neck=dict(
        type="CCN",
        output_dims=128,
        UAV_output_index=[3],  # 使用最后一个尺度作为UAV特征
        Satellite_ouput_index=[0, 1, 2, 3],  # 卫星使用所有尺度进行多尺度融合
    ),
    head=dict(
        type="CrossViewCrossAttention",
        dropout_rate=0.1,
        input_ndim=128,
        mid_ndim=256,  # 中间层的维度
        attention_layer_num=6,
    ),
    postprocess=dict(
        upsample_to_original=False,
    ),
    loss=dict(
        cls_loss=dict(type="GaussianFocalLoss", neg_weight=15, radius=3),
        smoothl1_loss=dict(type="SmoothL1Loss", reduction="mean"),
        haversine_loss=dict(type="HaversineLoss", reduction="mean"),
        weights=dict(
            cls_weight=1.0,
            smoothl1_weight=0.5,
            haversine_weight=0.5,
        ),
    ),
)


# 2. 数据集配置(datasets) =========================================
data_config = dict(
    batchsize=8,
    num_worker=8,
    val_batchsize=8,
    train_dir="/root/autodl-tmp/University-Release/train",
    val_dir="/root/autodl-tmp/University-Release/test",
    test_dir="/root/autodl-tmp/University-Release/test",
    test_mode="",  # University-Release测试集没有子目录模式
    UAVhw=[128, 128],
    Satellitehw=[384, 384],
)

pipline_config = dict(
    train_pipeline=dict(
        UAV=dict(
            RotateAndCrop=dict(rate=0.5),
            RandomAffine=dict(degrees=180),
            ColorJitter=dict(brightness=0.5, contrast=0.1, saturation=0.1, hue=0),
            RandomErasing=dict(probability=0.3),
            RandomResize=dict(img_size=data_config["UAVhw"]),
            ToTensor=dict(),
        ),
        Satellite=dict(
            ColorJitter=dict(brightness=0.5, contrast=0.1, saturation=0.1, hue=0),
            RandomErasing=dict(probability=0.3),
            RandomCrop=dict(cover_rate=0.85, map_size=(512, 1000)),
            RandomResize=dict(img_size=data_config["Satellitehw"]),
            ToTensor=dict(),
        ),
    ),
)

# 3. 训练策略配置(schedules) =========================================
lr_config = dict(lr=5e-5, type="cosine", warmup_iters=500, warmup_ratio=0.01)
train_config = dict(autocast=True, num_epochs=25)
test_config = dict(
    num_worker=8,
    filterR=1,
    checkpoint="/root/DRL/results/net_best.pth",  # 指向训练好的最佳模型
)

# 4. 运行配置(runtime) =========================================
checkpoint_config = dict(
    interval=1,
    epoch_start_save=6,
    only_save_best=True,
)
log_interval = 50
load_from = None
resume_from = None
debug = True
seed = 666
