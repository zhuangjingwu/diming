import torch
from .Dataloader_University import Dataloader_University, Sampler_University
import numpy as np


def train_collate_fn(batch):
    """
    修改collate_fn以匹配train.py的期望格式
    """
    img_s_list, img_d_list, ids_list = [], [], []
    
    for item in batch:
        img_s, _, img_d, idx, _, _, _ = item
        img_s_list.append(img_s)
        img_d_list.append(img_d)
        ids_list.append(idx)
    
    # 假设中心位置在图像中心
    ratex = torch.tensor([0.5] * len(batch))
    ratey = torch.tensor([0.5] * len(batch))
    
    return torch.stack(img_d_list, dim=0), torch.stack(img_s_list, dim=0), ratex, ratey





def make_dataset(opt,train=True):
    if train:
        # 准备transforms
        from torchvision import transforms
        
        # 使用配置文件中定义的图像大小
        uav_size = opt.data_config['UAVhw']
        satellite_size = opt.data_config['Satellitehw']
        
        # 构建transforms
        transform_drone = transforms.Compose([
            transforms.Resize(uav_size, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        transform_satellite = transforms.Compose([
            transforms.Resize(satellite_size, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        transform_train = {
            "satellite": transform_satellite,
            "train": transform_drone
        }
        
        image_datasets = Dataloader_University(root=opt.data_config.train_dir, transforms=transform_train, names=['satellite', 'drone'])
        sampler = Sampler_University(image_datasets, opt.data_config.batchsize)
        dataloaders = torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=opt.data_config.batchsize,
                                                 sampler=sampler,
                                                 num_workers=opt.data_config.num_worker,
                                                 pin_memory=True,
                                                 collate_fn=train_collate_fn
                                                 )
        dataset_sizes = {x: len(image_datasets) for x in ['satellite', 'drone']}
        return dataloaders, dataset_sizes

    else:
        # 准备transforms
        from torchvision import transforms
        
        # 使用配置文件中定义的图像大小
        uav_size = opt.data_config['UAVhw']
        satellite_size = opt.data_config['Satellitehw']
        
        # 构建transforms
        transform_drone = transforms.Compose([
            transforms.Resize(uav_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        transform_satellite = transforms.Compose([
            transforms.Resize(satellite_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        transform_val = {
            "satellite": transform_satellite,
            "train": transform_drone
        }
        
        dataset_test = Dataloader_University(root=opt.data_config.val_dir, transforms=transform_val, names=['satellite', 'drone'])
        
        # 使用自定义的val_collate_fn
        dataloaders = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=opt.data_config.val_batchsize,
                                                  shuffle=False,
                                                  num_workers=opt.data_config.num_worker,
                                                  pin_memory=True,
                                                  collate_fn=lambda batch: val_collate_fn(batch, opt))
        
        indices = np.random.choice(range(len(dataset_test)), 2000)
        dataset_test_subset = torch.utils.data.Subset(dataset_test, indices)
        dataloaders_sub = torch.utils.data.DataLoader(dataset_test_subset,
                                                  batch_size=opt.data_config.val_batchsize,
                                                  shuffle=False,
                                                  num_workers=opt.data_config.num_worker,
                                                  pin_memory=True,
                                                  collate_fn=lambda batch: val_collate_fn(batch, opt))
        
        return dataloaders, dataloaders_sub

# 修改val_collate_fn以接受opt参数
def val_collate_fn(batch, opt):
    """
    修改collate_fn以匹配train.py验证循环的期望格式
    """
    img_s_list, img_d_list, ids_list, satellite_paths, drone_paths = [], [], [], [], []
    
    for item in batch:
        img_s, _, img_d, idx, satellite_path, drone_path, _ = item
        img_s_list.append(img_s)
        img_d_list.append(img_d)
        ids_list.append(idx)
        satellite_paths.append(satellite_path)
        drone_paths.append(drone_path)
    
    # 假设中心位置在图像中心
    X = torch.tensor([opt.data_config.Satellitehw[0]//2] * len(batch))
    Y = torch.tensor([opt.data_config.Satellitehw[1]//2] * len(batch))
    
    return torch.stack(img_d_list, dim=0), torch.stack(img_s_list, dim=0), X, Y, drone_paths, satellite_paths




