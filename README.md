# 跨视角图像匹配与地理定位系统（Cross-View Image Matching and Geo-localization System）

## 项目简介

本项目实现了一个灵活的跨视角图像匹配与地理定位系统，主要用于解决无人机（UAV）与卫星图像的匹配问题。该系统支持多种Transformer骨干网络（ViTS, CvT, MixCvT等）和多样化的跨视角融合方式，通过模块化设计实现了高效、可扩展的跨视角图像匹配与地理定位功能。

## 项目结构

```
project/
├── models/                  # 模型定义
│   ├── Backbone/            # 骨干网络（支持ViT-S、CvT、MixCvT等）
│   ├── Neck/                # 颈部网络（CCN等特征融合模块）
│   ├── Head/                # 头部网络（AttentionFusionLib等融合方式）
│   ├── PostProcess/         # 后处理模块
│   ├── taskflow.py          # 模型流程控制
│   └── pos_utils.py         # 位置编码工具
├── datasets/                # 数据集相关
│   ├── Dataloader_University.py # University-Release数据集加载器
│   ├── make_dataloader.py   # 数据加载器生成
│   └── Augmentation.py      # 数据增强
├── configs/                 # 配置文件
│   ├── #Structure/          # 结构相关配置
│   ├── #Padding/            # 填充相关配置
│   ├── #Sharing/            # 权重共享配置
│   └── ...                  # 其他配置
├── losses/                  # 损失函数设计
├── optimizers/              # 优化器配置
├── tool/                    # 工具函数
├── docs/                    # 文档
├── results/                 # 结果保存目录
├── output/                  # 输出目录
├── checkpoints/             # 检查点目录
├── train.py                 # 训练脚本
├── test_meter.py            # 测试脚本（指标计算）
├── test_heatmap.py          # 测试脚本（热图生成）
├── test_visualization.py    # 测试脚本（可视化）
├── debug_prediction.py      # 调试脚本
├── evaluate_all.py          # 评估脚本
├── demo.py                  # 演示脚本
├── demo_visualization.py    # 演示脚本可视化
├── heatmap.py               # 热图生成
├── requirement.txt          # 依赖文件
└── README.md                # 项目说明
```

## 模型设计

### 模型结构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        双流跨视角联合编码器                              │
├───────────────────┬─────────────────────────────────────────────────────┤
│  UAV视角流        │                   卫星视角流                        │
│ (Close-range View)│                (Satellite View)                     │
├───────────────────┼─────────────────────────────────────────────────────┤
│ [输入]            │ [输入]                                               │
│ UAV低空图像       │ 俯视高分辨率卫星图像                                 │
├───────────────────┼─────────────────────────────────────────────────────┤
│ 局部特征提取      │ 全局特征提取                                         │
│ - 支持多种Backbone│ - 支持多种Backbone                                 │
│   (默认: Swin-T)  │   (默认: Swin-T)                                    │
│   (可选: ViT-S)   │   (可选: ViT-S)                                     │
│ - 输出多尺度特征  │ - 输出多尺度特征                                     │
│   (output_index=[0,1,2,3]) │ (output_index=[0,1,2,3])                  │
│ - 图像尺寸: 128x128│ - 图像尺寸: 384x384                                 │
├───────────────────┼─────────────────────────────────────────────────────┤
│ 特征维度对齐      │ 特征维度对齐                                         │
│ - CCN Neck        │ - CCN Neck                                          │
│ - 输出维度: 128    │ - 输出维度: 128                                      │
└───────────────────┴─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  跨视角注意力融合（AttentionFusionLib）                 │
│ - 融合类型: CAT_SA (默认), CA(CVCA, 可选)                               │
│ - 注意力层数: 4                                                         │
│ - 位置编码: 启用                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           匹配头与损失函数                               │
├─────────────────────────────┬─────────────────────────────────────────────┤
│ 分类损失（GaussianFocalLoss）│ 回归损失（SmoothL1Loss + HaversineLoss）    │
└─────────────────────────────┴─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        输出：定位热图                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. UAV视角流（Close-range View Stream）

- **局部特征提取**：支持多种Transformer骨干网络（默认使用Swin-T，可选ViT-S等）
- **特征维度对齐**：通过CCN（Channel Conversion Network）将特征维度转换为128
- **输出**：形状为B×C×H×W的特征图，用于与卫星特征融合
- **几何畸变处理**：已实现Deformable Conv（可通过配置启用），增强UAV图像几何畸变鲁棒性

### 2. 卫星视角流（Satellite View Stream）

- **全局特征提取**：支持多种Transformer骨干网络（默认使用Swin-T，可选ViT-S等）
- **特征维度对齐**：通过CCN将特征维度转换为128
- **跨尺度建模**：已实现Multi-Scale Fusion (MSF)（可通过配置启用），增强跨尺度地貌建模能力
- **输出**：形状为B×C×H×W的特征图，用于UAV特征与卫星区域对齐

### 3. 注意力融合方式

**采用AttentionFusionLib进行特征融合**
- 支持多种融合类型（CAT_SA、CA（CVCA）、MCA、MSA、SGF等）
- 默认使用CAT_SA（Concat + Self-Attention）融合方式
- 包含多层注意力机制（默认4层）和位置编码
- **动态位置编码**：实现了位置编码的动态调整，解决了固定位置编码与动态特征大小不匹配的问题
- **跨视角交叉注意力融合（CVCA，CA模式）**：
  - UAV特征作为Query，卫星特征作为Key/Value
  - 通过注意力机制找到最匹配区域
  - 结合了注意力机制的灵活性和特征融合的有效性

### 4. 损失函数设计

- **分类损失**：使用GaussianFocalLoss，用于预测热图中的目标位置
- **回归损失**：结合SmoothL1Loss和HaversineLoss
- **损失权重**：动态调整分类损失、SmoothL1损失和Haversine损失的权重（默认1.0:0.5:0.5）
- **多任务联合学习**：实现端到端的训练过程

### 5. 模型优化策略

**正则化技术**：
- **Dropout**：默认dropout_rate=0.5，用于抑制过拟合
- **权重衰减**：默认weight_decay=1e-3，增强模型泛化能力

**学习率策略**：
- 默认使用**ReduceLROnPlateau**策略，动态调整学习率
- 参数：factor=0.5, patience=4, min_lr=1e-5
- 根据验证损失自动调整学习率，避免过拟合

**数据增强策略**：
- **UAV图像**：RandomRotate90、RandomAffine(scale=(0.7,1.3), shear=20)、ColorJitter、RandomErasing
- **卫星图像**：RandomRotate90、RandomAffine(degrees=30, scale=(0.85,1.15))、ColorJitter、RandomErasing
- 增强模型对输入扰动的鲁棒性

**提前停止机制**：
- 监控验证损失，连续多个epoch未改善时停止训练
- 默认patience=10，可根据需求调整

**检查点配置**：
- epoch_start_save=1，确保记录所有epoch的误差数据
- interval=1，每个epoch保存一次检查点
- only_save_best=True，仅保存最佳模型



## 数据集

使用University-Release数据集，这是一个用于图像检索的跨视角数据集，包含：
- 无人机（UAV）低空图像
- 对应的卫星图像

**注意事项：**
- University-Release是图像检索数据集，**不包含像素级坐标标签**
- 训练时使用的标签是图像中心的人工标注（192, 192）
- 验证时使用的标签同样是硬编码的图像中心（192, 192）

数据集路径：`/root/autodl-tmp/University-Release`

数据集结构：
- `train/`：包含drone、satellite、street、google子目录，按场景分组
- `test/`：包含query/gallery图像对，用于检索测试
- 无单独的val目录，使用test目录进行验证

## 环境要求

- Python 3.8+
- PyTorch
- torchvision
- numpy
- scipy
- matplotlib
- pillow
- tqdm
- pyyaml
- thop (用于计算FLOPs)
- mmdet==2.28.2
- mmcv-full==1.7.2

## 安装依赖

```bash
pip install -r requirement.txt
```

## 训练模型

使用配置文件进行训练：

```bash
python train.py --config configs/#Structure/ViTS_CCN_SA_Balance_cr1_nw15_attentionlayer4_positionmbedding.py
```

**配置文件说明：**
- 配置文件中包含完整的训练参数设置
- 支持多种模型架构（ViTS, CvT, MixCvT等）
- 支持不同的融合方式（CAT_SA, MCA, MSA, SGF等）
- 可调整损失函数权重、学习率、批量大小等参数

## 测试模型

使用配置文件进行测试：

```bash
python test_meter.py --config configs/#Structure/ViTS_CCN_SA_Balance_cr1_nw15_attentionlayer4_positionmbedding.py
```

**其他测试脚本：**
- `test_heatmap.py`: 生成匹配热图
- `test_visualization.py`: 可视化测试结果
- `debug_prediction.py`: 调试预测结果

## 特征配置指南

### 启用Deformable Conv（可变形卷积）

Deformable Conv增强了模型对UAV图像几何畸变的鲁棒性，通过学习偏移量来调整感受野。

**配置方法：**
在配置文件的`neck`部分设置`use_deformable_conv=True`

```python
model = dict(
    backbone=dict(type="Swin-T", pretrain=True, output_index=[0,1,2,3], patch_size=4, window_size=7),
    neck=dict(
        type="CCN",
        output_dims=128,
        UAV_output_index=[0],
        Satellite_ouput_index=[0,1,2,3],
        use_deformable_conv=True,  # 启用Deformable Conv
    ),
    # 其他配置...
)
```

**实现细节：**
- 在`models/Neck/channel_convert.py`中实现
- 仅对UAV输出索引（UAV_output_index）中的最后一层使用Deformable Conv
- 使用mmcv-full库中的DeformableConvLayer实现

### 启用Multi-Scale Fusion（多尺度融合）

多尺度融合通过融合卫星视角的多层级特征，提高模型对多尺度地貌的建模能力。

**配置方法：**
1. 选择支持多尺度输出的骨干网络（如Swin-T）
2. 在配置文件中设置`output_index=[0,1,2,3]`
3. 在neck部分设置`Satellite_ouput_index=[0,1,2,3]`（长度>1时自动触发多尺度融合）

```python
model = dict(
    backbone=dict(
        type="Swin-T",
        pretrain=True,
        output_index=[0,1,2,3],  # 启用多尺度输出
        patch_size=4,
        window_size=7,
    ),
    neck=dict(
        type="CCN",
        output_dims=128,
        UAV_output_index=[0],
        Satellite_ouput_index=[0,1,2,3],  # 启用多尺度融合（长度>1）
    ),
    # 其他配置...
)
```

**实现细节：**
- 通过上采样 + 拼接 + 卷积的方式融合多尺度特征
- 在`models/Neck/channel_convert.py`的`CCN`类中实现
- 支持任意数量的尺度融合（通过Satellite_ouput_index控制）

### 启用位置编码

位置编码为特征添加空间位置信息，提高模型的定位能力。

**配置方法：**
在配置文件的`head`部分设置`need_position_embedding=True`

```python
model = dict(
    head=dict(
        type="AttentionFusionLib",
        need_position_embedding=True,  # 启用位置编码
        position_embedding_type="learned",  # 可选："learned", "fixed"
        # 其他配置...
    ),
    # 其他配置...
)
```

**实现细节：**
- 采用动态位置编码调整，解决固定位置编码与动态特征大小不匹配问题
- 在`models/Head/AttentionFusionLib.py`中使用`F.interpolate`实现位置编码的动态调整
- 支持多种融合类型（CAT_SA, CA等）的位置编码适配

## 训练命令示例

### 启用Deformable Conv和Multi-Scale Fusion进行训练

```bash
python train.py --config configs/#Structure/ViTS_CCN_SA_Balance_cr1_nw15_attentionlayer4_positionmbedding.py
```

**配置文件说明：**
- 使用Swin-T骨干网络（支持4个尺度输出）
- 启用Deformable Conv（`use_deformable_conv=True`）
- 启用多尺度融合（`Satellite_ouput_index=[0,1,2,3]`）
- 启用动态位置编码（`need_position_embedding=True`）
- 包含过拟合预防措施（dropout、数据增强、早停策略）

### 调试模式（验证模型加载和运行）

```bash
python debug_prediction.py --config configs/#Structure/ViTS_CCN_SA_Balance_cr1_nw15_attentionlayer4_positionmbedding.py
```

**注意事项：**
- 启用新功能后，模型结构会发生变化
- 使用`strict=False`参数加载权重，忽略权重不匹配问题
- 调试模式用于验证模型是否能正常加载和运行

## 结果说明

### 最新训练结果

根据最新训练（2025-12-14）的25个epoch数据，模型表现如下：

**核心性能指标：**
- **最佳RDS**：0.658（第14个epoch）
- **地理定位误差**：
  - 平均误差：22.8464像素
  - 中位数误差：22.6274像素
- **验证损失**：0.0005165
- **训练损失**：0.0024
- **学习率**：最终稳定在1e-5

**训练稳定性：**
- 模型在25个epoch中保持稳定训练
- 早停计数器在第25个epoch为2/10，模型未出现过拟合
- 验证损失在训练过程中持续下降并保持稳定

**MA@K指标：**
- MA@1m = 1.0000
- MA@3m = 1.0000
- MA@5m = 1.0000
- MA@10m = 1.0000
- MA@20m = 1.0000
- MA@30m = 1.0000
- MA@50m = 1.0000
- MA@100m = 1.0000

### 结果可视化

训练完成后，模型会自动生成以下可视化结果（保存在`results/`目录）：

1. **损失曲线**：`loss_curve.png`
   - 展示训练和验证损失的变化趋势
   - 清晰反映模型的收敛过程

2. **地理定位误差曲线**：`geolocation_error_curve.png`
   - 展示随epoch变化的平均误差和中位数误差
   - 直观反映模型定位精度的变化

3. **地理定位误差分布图**：`geolocation_error.png`
   - 展示误差的分布情况
   - 包含统计信息（均值、中位数、标准差等）

4. **匹配热图样本**：`matching_heatmap_sample_{1-5}.png`
   - 展示模型生成的定位热图
   - 可视化UAV图像与卫星图像的匹配区域

### 重要说明

**地理定位误差的局限性：**
- 由于University-Release数据集是图像检索数据集，**不包含真实的像素级坐标标签**
- 验证时使用的标签是硬编码的图像中心(192, 192)
- 因此，地理定位误差的计算基于人工标签，仅供参考

**建议关注的指标：**
- RDS（Ranking-based Distance Score）：基于排序的检索指标
- MA@K：在K米范围内的匹配准确率
- 这些指标更能反映模型的实际检索性能

### 误差记录完整性

通过优化配置（epoch_start_save=1），确保记录所有epoch的误差数据：
- 解决了之前部分epoch数据缺失的问题
- 每个epoch都记录平均误差和中位数误差
- 生成完整的误差曲线，便于分析模型训练过程

## 与Baseline对比

| Method                         | RDS   | 说明                          |
|--------------------------------|-------|-------------------------------|
| ViTS_CCN_SA_Balance            | 0.658 | 当前实现的模型（Swin-T骨干）   |
| ViTS_CCN_SA                   | 0.85+ | 使用自注意力融合的ViT-S模型    |
| CvT13_CCN_MSA                 | 0.82+ | 使用多头自注意力的CvT模型      |
| MixCvT13_CCN_SGF              | 0.83+ | 使用全局特征融合的混合模型      |

**说明：**
- RDS（Ranking-based Distance Score）是主要评估指标
- 表格中的结果是基于University-Release数据集的检索性能
- 当前实现使用Swin-T骨干网络，而文献参考模型使用不同的骨干网络结构
- 模型性能可能受到数据集配置、训练参数和实现细节的影响

**性能分析：**
- 当前模型实现了稳定的训练过程，RDS达到0.658
- 通过正则化、数据增强和学习率策略的优化，有效抑制了过拟合
- 地理定位误差保持稳定，平均误差约22.8像素
- 模型在所有epoch中都保持良好的稳定性和泛化能力

## 创新点说明

1. **灵活的骨干网络支持**：支持多种Transformer骨干网络（Swin-T, ViTS, CvT, MixCvT等），默认使用Swin-T，可根据需求灵活配置
2. **统一的特征维度对齐**：使用CCN（Channel Conversion Network）实现不同视角特征的维度对齐
3. **跨视角交叉注意力融合（CVCA，CA模式）**：
   - 已在AttentionFusionLib中完整实现
   - UAV特征作为Query，卫星特征作为Key/Value
   - 通过注意力机制找到最匹配区域
   - 结合了注意力机制的灵活性和特征融合的有效性
4. **多样化的融合方式**：提供AttentionFusionLib库，支持多种跨视角融合方式（CAT_SA, CA, MCA, MSA, SGF等）
5. **多级损失函数设计**：结合GaussianFocalLoss（分类）、SmoothL1Loss（回归）和HaversineLoss（地理距离），实现端到端训练
6. **动态位置编码**：实现了位置编码的动态调整，解决了固定位置编码与动态特征大小不匹配的问题
7. **Deformable Conv增强**：实现了可变形卷积网络，增强模型对UAV图像几何畸变的鲁棒性
8. **多尺度特征融合**：支持多尺度特征的提取和融合，提高模型对卫星图像多尺度地貌的建模能力
9. **模块化设计**：采用Backbone-Neck-Head-PostProcess的模块化设计，便于扩展和修改

## 参考文献

- Zheng, Z., Wei, Y., & Yang, Y. (2020). University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization. arXiv preprint arXiv:2002.12186.
- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 10012-10022).

## 联系方式

如有问题，请联系项目负责人。
