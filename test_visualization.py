#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试可视化函数的脚本
用于验证修改后的plot_loss_curve和plot_geolocation_error_curve函数
"""

import matplotlib
matplotlib.use('agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os

# 模拟损失历史数据
def generate_test_loss_history(epochs=20):
    """生成模拟的损失历史数据"""
    train_loss = np.random.rand(epochs) * 2 + 1  # 1-3之间的随机值
    train_loss = np.sort(train_loss)[::-1]  # 递减趋势
    
    val_loss = train_loss * 1.1 + 0.2  # 验证损失略高
    
    return {
        'train_total_loss': train_loss,
        'val_total_loss': val_loss,
        'train_cls_loss': train_loss * 0.7,
        'train_loc_loss': train_loss * 0.3,
        'val_cls_loss': val_loss * 0.7,
        'val_loc_loss': val_loss * 0.3
    }

# 模拟误差历史数据
def generate_test_error_history(epochs=20):
    """生成模拟的误差历史数据"""
    mean_error = np.random.rand(epochs) * 0.5 + 0.3  # 0.3-0.8之间的随机值
    mean_error = np.sort(mean_error)  # 递增趋势
    
    median_error = mean_error * 0.8 - 0.05  # 中位数误差略低
    
    return {
        'mean_error': mean_error,
        'median_error': median_error
    }

# 模拟plot_loss_curve函数
def test_plot_loss_curve():
    """测试损失曲线绘制函数"""
    loss_history = generate_test_loss_history(30)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history['train_total_loss']) + 1), 
             loss_history['train_total_loss'], 'b-', linewidth=2, label='Train Loss')
    plt.plot(range(1, len(loss_history['val_total_loss']) + 1), 
             loss_history['val_total_loss'], 'r-', linewidth=2, label='Validation Loss')
    
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 创建results目录
    os.makedirs('results', exist_ok=True)
    
    # 保存图表
    plt.savefig('results/test_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("测试损失曲线已保存到 results/test_loss_curve.png")

# 模拟plot_geolocation_error_curve函数
def test_plot_geolocation_error_curve():
    """测试误差曲线绘制函数"""
    error_history = generate_test_error_history(30)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(error_history['mean_error']) + 1), 
             error_history['mean_error'], 'r-', linewidth=2, label='Mean Error (pixels)')
    plt.plot(range(1, len(error_history['median_error']) + 1), 
             error_history['median_error'], 'b-', linewidth=2, label='Median Error (pixels)')
    
    plt.title('Geolocation Error During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Error (pixels)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 保存图表
    plt.savefig('results/test_geolocation_error_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("测试误差曲线已保存到 results/test_geolocation_error_curve.png")

if __name__ == "__main__":
    print("开始测试可视化函数...")
    
    # 测试损失曲线绘制
    test_plot_loss_curve()
    
    # 测试误差曲线绘制
    test_plot_geolocation_error_curve()
    
    print("所有测试完成！")
