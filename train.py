# -*- coding: utf-8 -*-
import argparse
import torch
import os

# 设置Hugging Face镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = 'False'

from torch.autograd import Variable
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
from torch.cuda.amp import autocast
from models.taskflow import make_model
from datasets.make_dataloader import make_dataset
from losses.make_loss import make_loss
from tool.utils_server import calc_flops_params, save_network, copyfiles2checkpoints, get_logger, TensorBoardManager
from tool.evaltools import evaluate
from tqdm import tqdm
import numpy as np
import cv2
import random
import json
from collections import defaultdict
from tool.evaltools import Distance
from mmcv import Config
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image


def create_hanning_mask(center_R):
    hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
        np.hanning(center_R+2),
        np.hanning(center_R+2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1, 1:-1]


def get_config():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument(
        '--config', default='configs/#Structure/ViTS_CCN_SA_Balance_cr1_nw15_attentionlayer4_positionmbedding.py',
        type=str, help='config filename')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default="test",
                        type=str, help='output model name')
    opt = parser.parse_args()
    if opt.name == "":
        opt.name = opt.config.split("/")[-1].split(".py")[0].split("configs/")[-1]
    print(opt.name)
    cfg = Config.fromfile(opt.config)
    for key, value in cfg.items():
        setattr(opt, key, value)
    return opt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    random.seed(seed)

def setup_device(opt):
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        # cudnn.benchmark = True


def train_model(model, loss_func, opt, dataloaders, dataset_sizes):
    use_gpu = opt.use_gpu
    num_epochs = opt.train_config["num_epochs"]
    output_dir = os.path.join("checkpoints", opt.name, "output")
    os.makedirs(output_dir, exist_ok=True)
    cur_time = datetime.datetime.now()
    logger_file = os.path.join(output_dir, "train_{}.log".format(cur_time))
    logger = get_logger(logger_file)
    # init tensorboard writer
    tensorboard_writer = TensorBoardManager(
        os.path.join(output_dir, "summary"))
    
    macs, params = calc_flops_params(
        model, (1, 3, opt.data_config['UAVhw'][0], opt.data_config['UAVhw'][1]), (1, 3, opt.data_config['Satellitehw'][0], opt.data_config['Satellitehw'][1]))
    logger.info("MACs={}, Params={}".format(macs, params))

    since = time.time()

    scaler = GradScaler()

    best_RDS = 0

    logger.info('start training!')

    optimizer, scheduler = make_optimizer(model, opt)

    # 初始化损失记录
    loss_history = {
        'train_total_loss': [],
        'train_cls_loss': [],
        'train_loc_loss': [],
        'val_total_loss': [],
        'val_cls_loss': [],
        'val_loc_loss': []
    }
    
    # 初始化误差记录（随epoch变化）
    error_history = {
        'mean_error': [],
        'median_error': []
    }
    
    # 初始化早停相关变量
    best_val_loss = float('inf')
    patience = 3  # 连续3个epoch验证损失没有改善就停止
    patience_counter = 0
    early_stopping = False
    
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        logger.info('-' * 50)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_cls_loss = 0.0
        running_loc_loss = 0.0
        iter_cls_loss = 0.0
        iter_loc_loss = 0.0
        iter_start = time.time()
        iter_loss = 0
        total_iters = len(dataloaders["train"])
        # train
        for iter, (z, x, ratex, ratey) in enumerate(dataloaders["train"]):
            now_batch_size, _, _, _ = z.shape

            if now_batch_size < opt.data_config["batchsize"]:  # skip the last batch
                continue
            if use_gpu:
                z = Variable(z.cuda().detach())
                x = Variable(x.cuda().detach())
            else:
                z, x = Variable(z), Variable(x)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # start_time = time.time()
            # if opt.train_config["autocast"]:
            #     with autocast():
            #         outputs = model(z, x)  # satellite and drone
            # else:
            outputs = model(z, x)
            # print("model_time:{}".format(time.time()-start_time))
            total_loss, loss_dict = loss_func(outputs, [ratex, ratey])
            loss = total_loss
            # 从loss_dict中提取损失用于日志记录
            cls_loss = loss_dict.get("cls_loss", 0.0)
            loc_loss = loss_dict.get("smoothl1_loss", 0.0) + loss_dict.get("haversine_loss", 0.0)
            # backward + optimize only if in training phase
            loss_backward = loss
            # start_time = time.time()
            if opt.train_config["autocast"]:
                scaler.scale(loss_backward).backward()
                scaler.step(optimizer)
                scaler.update()
                # 只有非ReduceLROnPlateau调度器才在每个迭代步骤中调用step
                if opt.lr_config["type"] != "ReduceLROnPlateau":
                    scheduler.step()
            else:
                loss_backward.backward()
                optimizer.step()
                # 只有非ReduceLROnPlateau调度器才在每个迭代步骤中调用step
                if opt.lr_config["type"] != "ReduceLROnPlateau":
                    scheduler.step()
            # print("loss_backward_time:{}".format(time.time()-start_time))

            # statistics
            running_loss += loss.item() * now_batch_size
            running_cls_loss += cls_loss * now_batch_size
            running_loc_loss += loc_loss * now_batch_size
            iter_loss += loss.item() * now_batch_size
            iter_cls_loss += cls_loss * now_batch_size
            iter_loc_loss += loc_loss * now_batch_size

            if (iter + 1) % opt.log_interval == 0:
                time_elapsed_part = time.time() - iter_start
                iter_loss = iter_loss/opt.log_interval/now_batch_size
                iter_cls_loss = iter_cls_loss/opt.log_interval/now_batch_size
                iter_loc_loss = iter_loc_loss/opt.log_interval/now_batch_size

                lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']

                tensorboard_writer.add_scalar(
                    "loss/total_loss", iter_loss, epoch*total_iters+iter)
                tensorboard_writer.add_scalar(
                    "loss/cls_loss", iter_cls_loss, epoch*total_iters+iter)
                tensorboard_writer.add_scalar(
                    "loss/loc_loss", iter_loc_loss, epoch*total_iters+iter)
                tensorboard_writer.add_scalar(
                    "lr", lr_backbone, epoch*total_iters+iter)

                logger.info("[{}/{}] loss: {:.4f} cls_loss: {:.4f} loc_loss:{:.4f} lr_backbone:{:.6f} time:{:.0f}m {:.0f}s ".format(
                    iter + 1, total_iters, iter_loss, iter_cls_loss, iter_loc_loss, lr_backbone, time_elapsed_part // 60, time_elapsed_part % 60))
                iter_loss = 0.0
                iter_loc_loss = 0.0
                iter_cls_loss = 0.0
                iter_start = time.time()

        epoch_loss = running_loss / dataset_sizes['satellite']
        epoch_cls_loss = running_cls_loss / dataset_sizes['satellite']
        epoch_loc_loss = running_loc_loss / dataset_sizes['satellite']
        
        # 记录训练损失
        loss_history['train_total_loss'].append(epoch_loss)
        loss_history['train_cls_loss'].append(epoch_cls_loss)
        loss_history['train_loc_loss'].append(epoch_loc_loss)

        lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']

        time_elapsed = time.time() - since
        logger.info('Epoch[{}/{}] Loss: {:.4f}  lr_backbone:{:.6f}  time:{:.0f}m {:.0f}s'.format(
            epoch+1, num_epochs, epoch_loss, lr_backbone, time_elapsed // 60, time_elapsed % 60))

        # ----------------------save and test the model------------------------------ #
        # 每个epoch都进行验证，收集误差数据
        model.eval()
        total_score = 0.0
        total_score_b = 0.0
        start_time = time.time()
        flag_bias = 0
        MA_json_save = []
        MA_dict = defaultdict(int)
        MA_log_list = [1, 3, 5, 10, 20, 30, 50, 100]
        tensorboard_image_ind = 0
        val_loss = 0.0
        val_cls_loss = 0.0
        val_loc_loss = 0.0
        sample_nums = 0
        
        # 用于存储地理定位误差
        geolocation_errors = []
        
        # 选择验证数据集
        val_dataloader = dataloaders["val"] if ((epoch + 1)-opt.checkpoint_config["epoch_start_save"]) % opt.checkpoint_config["interval"] == 0 and (epoch+1) >= opt.checkpoint_config["epoch_start_save"] or (epoch+1 == opt.train_config["num_epochs"]) else dataloaders["val_sub"]
        
        for uav, satellite, X, Y, uav_path, satellite_path in tqdm(val_dataloader):
            sample_nums += uav.shape[0]
            z = uav.cuda()
            x = satellite.cuda()
            rate_x = X/opt.data_config["Satellitehw"][0]
            rate_y = Y/opt.data_config["Satellitehw"][1]
            with torch.no_grad():
                response, loc_bias = model(z, x)
                total_loss, loss_dict = loss_func([response, loc_bias], [rate_x, rate_y])
                cls_loss = loss_dict.get("cls_loss", 0.0)
                loc_loss = loss_dict.get("smoothl1_loss", 0.0) + loss_dict.get("haversine_loss", 0.0)
            val_iter_loss = total_loss
            val_loss += val_iter_loss/len(val_dataloader)
            val_cls_loss += cls_loss/len(val_dataloader)
            val_loc_loss += loc_loss/len(val_dataloader)

            if opt.model["loss"]["cls_loss"].get("use_softmax", False):
                response = torch.softmax(response,dim=1)[:,1:]
            else:
                response = torch.sigmoid(response)
            maps = response.squeeze().cpu().detach().numpy()
            # 遍历每一个batch
            for ind, map in enumerate(maps):
                if opt.test_config["filterR"] != 1:
                    kernel = create_hanning_mask(opt.test_config["filterR"])
                    map = cv2.filter2D(map, -1, kernel)

                label_XY = np.array(
                    [X[ind].squeeze().detach().numpy(), Y[ind].squeeze().detach().numpy()])

                satellite_map = cv2.resize(map, opt.data_config["Satellitehw"])
                id = np.argmax(satellite_map)
                S_X = int(id // opt.data_config["Satellitehw"][0])
                S_Y = int(id % opt.data_config["Satellitehw"][1])

                # 计算地理定位误差（使用图像坐标的欧几里得距离作为替代）
                pred_XY = np.array([S_X, S_Y])
                error = np.sqrt(np.sum((pred_XY - label_XY) ** 2))
                geolocation_errors.append(error)

                # 获取预测的经纬度信息 - University-Release数据集没有GPS信息，使用占位符
                get_gps_x = S_X / opt.data_config["Satellitehw"][0]
                get_gps_y = S_Y / opt.data_config["Satellitehw"][0]
                
                # 由于University-Release数据集没有提供GPS信息，跳过MA指标计算
                meter_distance = 0.0
                MA_json_save.append(meter_distance)
                for meter in MA_log_list:
                    if meter_distance <= meter:
                        MA_dict[meter] += 1

                # 统计RDS指标
                single_score = evaluate(
                    opt, pred_XY=pred_XY, label_XY=label_XY)
                total_score += single_score
                if loc_bias is not None:
                        flag_bias = 1
                        loc = loc_bias.squeeze().cpu().detach().numpy()
                        id_map = np.argmax(map)
                        S_X_map = int(id_map // map.shape[-1])
                        S_Y_map = int(id_map % map.shape[-1])
                        pred_XY_map = np.array([S_X_map, S_Y_map])
                        # loc的形状是(B, 2, H_x, W_x)，所以正确的索引方式是[:, :, S_X_map, S_Y_map]
                        pred_XY_b = (
                            pred_XY_map + loc[:, :, S_X_map, S_Y_map]) * opt.data_config["Satellitehw"][0] / loc.shape[-1]  # add bias
                        pred_XY_b = np.array(pred_XY_b)
                        # 由于是单个样本，取出第一个元素
                        pred_XY_b = pred_XY_b[0]
                        single_score_b = evaluate(
                            opt, pred_XY=pred_XY_b, label_XY=label_XY)
                        total_score_b += single_score_b

                # print("pred: " + str(pred_XY) + " label: " +str(label_XY) +" score:{}".format(single_score))
                # TODO:将可视化图像添加到tensorboard中
        
        # 在训练结束时保存地理定位误差
        if epoch+1 == opt.train_config["num_epochs"]:
            opt.geolocation_errors = geolocation_errors

        # time
        time_consume = time.time() - start_time
        logger.info("time consume is {}".format(time_consume))

        # total loss
        logger.info("valset total loss is {}".format(val_loss))
        
        # 记录验证损失
        loss_history['val_total_loss'].append(float(val_loss))
        loss_history['val_cls_loss'].append(float(val_cls_loss))
        loss_history['val_loc_loss'].append(float(val_loc_loss))

        # 计算并记录该epoch的平均误差和中位数误差
        if geolocation_errors:
            mean_error = np.mean(geolocation_errors)
            median_error = np.median(geolocation_errors)
            error_history['mean_error'].append(mean_error)
            error_history['median_error'].append(median_error)
            logger.info(f"Epoch {epoch+1} - Mean Error: {mean_error:.4f}, Median Error: {median_error:.4f}")
        
        # 早停逻辑检查
        current_val_loss = float(val_loss)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"Early stopping counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            early_stopping = True
            break
        
        # 对于ReduceLROnPlateau调度器，在验证阶段基于验证损失调用step
        if opt.lr_config["type"] == "ReduceLROnPlateau":
            scheduler.step(current_val_loss)
        
        # 只有在满足检查点条件时才保存模型和计算详细指标
        if ((epoch + 1)-opt.checkpoint_config["epoch_start_save"]) % opt.checkpoint_config["interval"] == 0 and (epoch+1) >= opt.checkpoint_config["epoch_start_save"] or (epoch+1 == opt.train_config["num_epochs"]):
            # if "only_save_best" is False， save the checkpoint
            if not opt.checkpoint_config["only_save_best"]:
                save_name = "last" if epoch+1 == opt.train_config["num_epochs"] else epoch+1
                save_network(model, opt.name, save_name)
            
            # 计算RDS指标
            RDS = total_score / sample_nums
            # save the best checkpoint
            if RDS > best_RDS:
                best_RDS = RDS
                best_epoch = epoch+1
                save_network(model, opt.name, "best")
            logger.info("Epoch{}: the RDS is {}".format(epoch+1, RDS))
            if flag_bias:
                RDS_b = total_score_b / sample_nums
                logger.info(
                    "Epoch{}: the bias RDS is {}".format(epoch+1, RDS_b))

            # MA@K
            for log_meter in MA_log_list:
                logger.info("MA@{}m = {:.4f}".format(log_meter,
                            MA_dict[log_meter]/sample_nums))
        
        if early_stopping:
            break
    logger.info("saved best epoch is {}, RDS is {:.3f}".format(best_epoch, best_RDS))
    
    # 绘制并保存损失曲线
    def plot_loss_curve(loss_history, save_path):
        epochs = range(1, len(loss_history['train_total_loss']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss_history['train_total_loss'], 'b-', linewidth=2, label='Train Loss')
        plt.plot(epochs, loss_history['val_total_loss'], 'r-', linewidth=2, label='Validation Loss')
        
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Loss curve saved to {save_path}")
    
    # 创建results目录
    results_dir = '/root/DRL/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存损失曲线
    loss_curve_path = os.path.join(results_dir, 'loss_curve.png')
    plot_loss_curve(loss_history, loss_curve_path)
    
    # 绘制并保存地理定位误差分布图
    def plot_geolocation_error(errors, save_path):
        plt.figure(figsize=(10, 6))
        
        # 绘制直方图
        n, bins, patches = plt.hist(errors, bins=50, density=True, alpha=0.7, color='b')
        
        # 添加统计信息
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        
        plt.axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f}')
        plt.axvline(median_error, color='g', linestyle='--', label=f'Median: {median_error:.2f}')
        
        plt.title('Geolocation Error Distribution')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        
        # 在图表上添加统计信息
        stats_text = f'STD: {std_error:.2f}\nMax: {np.max(errors):.2f}\nMin: {np.min(errors):.2f}'
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top', horizontalalignment='right', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Geolocation error distribution saved to {save_path}")
    
    # 保存地理定位误差分布图
    if hasattr(opt, 'geolocation_errors') and opt.geolocation_errors:
        geolocation_error_path = os.path.join(results_dir, 'geolocation_error.png')
        plot_geolocation_error(opt.geolocation_errors, geolocation_error_path)
    
    # 绘制并保存随epoch变化的地理定位误差曲线
    def plot_geolocation_error_curve(error_history, save_path):
        epochs = range(1, len(error_history['mean_error']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, error_history['mean_error'], 'r-', linewidth=2, label='Mean Error (pixels)')
        plt.plot(epochs, error_history['median_error'], 'b-', linewidth=2, label='Median Error (pixels)')
        
        plt.title('Geolocation Error During Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Error (pixels)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Geolocation error curve saved to {save_path}")
    
    # 保存随epoch变化的地理定位误差曲线
    geolocation_error_curve_path = os.path.join(results_dir, 'geolocation_error_curve.png')
    plot_geolocation_error_curve(error_history, geolocation_error_curve_path)
    
    # 生成匹配热图样本
    def generate_heatmap_samples(opt, dataloader, model, results_dir, num_samples=5):
        # 归一化函数
        def normalization(data):
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range
        
        # 生成热图
        def gen_heatmap(heatmap, ori_image):
            # 确保heatmap是单通道
            if heatmap.ndim == 3 and heatmap.shape[0] == 1:
                heatmap = heatmap[0]
            elif heatmap.ndim == 3 and heatmap.shape[0] > 1:
                # 如果是多通道，取第一个通道
                heatmap = heatmap[0]
            elif heatmap.ndim == 3 and heatmap.shape[-1] > 1:
                # 如果是HWC格式的多通道，取第一个通道
                heatmap = heatmap[:, :, 0]
            
            heatmap = normalization(heatmap)
            heatmap = cv2.resize(heatmap, (ori_image.shape[1], ori_image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, 2)
            superimposed_img = heatmap * 0.5 + ori_image * 0.5
            return superimposed_img.astype(np.uint8)
        
        model.eval()
        sample_count = 0
        
        for uav, satellite, X, Y, uav_path, satellite_path in dataloader:
            if sample_count >= num_samples:
                break
                
            z = uav.cuda()
            x = satellite.cuda()
            
            with torch.no_grad():
                response, _ = model(z, x)
                
            if opt.model["loss"]["cls_loss"].get("use_softmax", False):
                response = torch.softmax(response, dim=1)[:, 1:]
            else:
                response = torch.sigmoid(response)
                
            map = response.squeeze().cpu().detach().numpy()
            
            if opt.test_config["filterR"] != 1:
                kernel = create_hanning_mask(opt.test_config["filterR"])
                map = cv2.filter2D(map, -1, kernel)
                
            # 加载原始图像
            uav_img = cv2.imread(uav_path[0])
            satellite_img = cv2.imread(satellite_path[0])
            
            # 调整图像大小
            uav_img = cv2.resize(uav_img, opt.data_config["UAVhw"])
            satellite_img = cv2.resize(satellite_img, opt.data_config["Satellitehw"])
            
            # 生成热图
            heatmap = gen_heatmap(map, satellite_img)
            
            # 获取预测和真实位置
            satellite_map = cv2.resize(map, opt.data_config["Satellitehw"])
            id = np.argmax(satellite_map)
            pred_X = int(id // opt.data_config["Satellitehw"][0])
            pred_Y = int(id % opt.data_config["Satellitehw"][1])
            
            label_X = int(X[0].squeeze().detach().numpy())
            label_Y = int(Y[0].squeeze().detach().numpy())
            
            # 在热图上标记预测和真实位置
            heatmap = cv2.circle(heatmap, (pred_Y, pred_X), radius=5, color=(255, 0, 0), thickness=3)
            heatmap = cv2.circle(heatmap, (label_Y, label_X), radius=5, color=(0, 255, 0), thickness=3)
            
            # 创建组合图像
            combined_img = np.zeros((max(uav_img.shape[0], heatmap.shape[0]), uav_img.shape[1] + heatmap.shape[1], 3), dtype=np.uint8)
            combined_img[:uav_img.shape[0], :uav_img.shape[1]] = uav_img
            combined_img[:heatmap.shape[0], uav_img.shape[1]:] = heatmap
            
            # 添加标题
            cv2.putText(combined_img, "UAV Image", (uav_img.shape[1]//4, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_img, "Satellite Image with Heatmap", (uav_img.shape[1] + heatmap.shape[1]//4, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 保存组合图像
            heatmap_path = os.path.join(results_dir, f'matching_heatmap_sample_{sample_count+1}.png')
            cv2.imwrite(heatmap_path, combined_img)
            logger.info(f"Matching heatmap sample saved to {heatmap_path}")
            
            sample_count += 1
    
    # 生成匹配热图样本
    logger.info("Generating matching heatmap samples...")
    generate_heatmap_samples(opt, dataloaders["val"], model, results_dir, num_samples=5)




if __name__ == '__main__':
    opt = get_config()

    # init device
    setup_device(opt)

    # init seed
    setup_seed(opt.seed)
    
    # init dataloader
    dataloaders_train, dataset_sizes = make_dataset(opt)
    dataloaders_val, dataloaders_val_sub = make_dataset(opt, train=False)
    dataloaders = {"train": dataloaders_train,
                   "val": dataloaders_val,
                   "val_sub": dataloaders_val_sub}
    opt.train_iters_per_epoch = len(dataloaders["train"])

    # init model
    model = make_model(opt)
    model = model.cuda()
    
    # init loss
    loss_func = make_loss(opt)
    # copy current demos to a seperate dir
    copyfiles2checkpoints(opt)
    # train the model
    train_model(model, loss_func, opt, dataloaders, dataset_sizes)
