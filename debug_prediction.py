# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('/root/DRL')
import torch
import numpy as np
from mmcv import Config
from models.taskflow import make_model
from datasets.make_dataloader import make_dataset
import cv2
import matplotlib.pyplot as plt

# 创建hanning掩码函数
def create_hanning_mask(center_R):
    hann_window = np.outer(
        np.hanning(center_R+2),
        np.hanning(center_R+2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1, 1:-1]

# 获取配置
config_file = '/root/DRL/configs/#Structure/ViTS_CCN_SA_Balance_cr1_nw15_attentionlayer4_positionmbedding.py'
cfg = Config.fromfile(config_file)

# 设置设备
torch.cuda.set_device(0)

# 创建模型
model = make_model(cfg)
model.cuda()
model.eval()

# 加载预训练权重
checkpoint_path = '/root/DRL/results/net_best.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint, strict=False)
    print("Loaded checkpoint with mismatched layers (expected when enabling new features)")
else:
    print("Checkpoint not found, using random weights")

# 创建验证数据集
dataloader_val, dataloader_val_sub = make_dataset(cfg, train=False)
# 使用小批量验证集进行快速测试
if dataloader_val_sub is not None:
    dataloader_val = dataloader_val_sub

# 用于存储所有预测结果
all_pred_XY = []
all_label_XY = []
all_errors = []

# 推理
with torch.no_grad():
    for batch_idx, (uav, satellite, X, Y, uav_path, satellite_path) in enumerate(dataloader_val):
        if batch_idx >= 2:  # 只处理前2个批次
            break
            
        z = uav.cuda()
        x = satellite.cuda()
        
        # 前向传播
        output = model(z, x)
        # 获取特征图（output是元组，第一个元素是特征图）
        cls_out = output[0]
        
        # 处理每个样本
        for ind in range(z.shape[0]):
            # 获取输出特征图
            map = cls_out[ind].squeeze().detach().cpu().numpy()
            
            # 应用hanning滤波器（如果配置了）
            if hasattr(cfg, 'test_config') and 'filterR' in cfg.test_config:
                kernel = create_hanning_mask(cfg.test_config['filterR'])
                map = cv2.filter2D(map, -1, kernel)
            
            # 获取真实标签（硬编码为中心）
            label_XY = np.array([X[ind].squeeze().detach().numpy(), Y[ind].squeeze().detach().numpy()])
            
            # 调整特征图大小并找到最大值位置
            satellite_map = cv2.resize(map, cfg.data_config['Satellitehw'])
            id = np.argmax(satellite_map)
            S_X = int(id // cfg.data_config['Satellitehw'][0])
            S_Y = int(id % cfg.data_config['Satellitehw'][1])
            
            # 计算误差
            pred_XY = np.array([S_X, S_Y])
            error = np.sqrt(np.sum((pred_XY - label_XY) ** 2))
            
            # 存储结果
            all_pred_XY.append(pred_XY)
            all_label_XY.append(label_XY)
            all_errors.append(error)
            
            # 打印信息
            print(f"Sample {ind+1} in batch {batch_idx+1}:")
            print(f"  UAV path: {uav_path[ind]}")
            print(f"  Satellite path: {satellite_path[ind]}")
            print(f"  Predicted position: ({S_X}, {S_Y})")
            print(f"  Label position: ({label_XY[0]}, {label_XY[1]})")
            print(f"  Error: {error:.2f} pixels")
            print(f"  Output map shape: {map.shape}, Resized map shape: {satellite_map.shape}")
            print(f"  Max value in map: {np.max(satellite_map):.4f}, Min value: {np.min(satellite_map):.4f}")
            print()

# 分析结果
print("="*50)
print("Analysis Results:")
print(f"Number of samples processed: {len(all_pred_XY)}")
print(f"All predicted positions:")
for i, pred in enumerate(all_pred_XY):
    print(f"  Sample {i+1}: ({pred[0]}, {pred[1]})")

print(f"\nAll errors:")
for i, error in enumerate(all_errors):
    print(f"  Sample {i+1}: {error:.2f}")

print(f"\nError statistics:")
print(f"  Mean error: {np.mean(all_errors):.4f}")
print(f"  Std error: {np.std(all_errors):.4f}")
print(f"  Max error: {np.max(all_errors):.4f}")
print(f"  Min error: {np.min(all_errors):.4f}")

# 检查是否所有预测位置都相同
unique_preds = np.unique(np.array(all_pred_XY), axis=0)
print(f"\nNumber of unique predictions: {len(unique_preds)}")
if len(unique_preds) == 1:
    print(f"All predictions are the same: {unique_preds[0]}")

# 绘制预测位置分布
plt.figure(figsize=(10, 10))
plt.xlim(0, cfg.data_config['Satellitehw'][0])
plt.ylim(0, cfg.data_config['Satellitehw'][1])

# 绘制所有预测位置
pred_X = [p[0] for p in all_pred_XY]
pred_Y = [p[1] for p in all_pred_XY]
plt.scatter(pred_X, pred_Y, color='red', label='Predictions', s=100, alpha=0.7)

# 绘制真实标签位置
label_X = [l[0] for l in all_label_XY]
label_Y = [l[1] for l in all_label_XY]
plt.scatter(label_X, label_Y, color='green', label='Labels (Center)', s=100, alpha=0.7)

# 绘制误差圆
circle = plt.Circle((192, 192), 24.04, color='blue', fill=False, linestyle='--', label='Error Circle (24.04 pixels)')
plt.gca().add_patch(circle)

plt.xlabel('X Coordinate (pixels)')
plt.ylabel('Y Coordinate (pixels)')
plt.title('Prediction Positions vs Label Positions')
plt.legend()
plt.grid(True)
plt.savefig('/root/DRL/prediction_distribution.png', dpi=300, bbox_inches='tight')
print("\nPrediction distribution plot saved to /root/DRL/prediction_distribution.png")
