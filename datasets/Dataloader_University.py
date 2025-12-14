import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image

class Dataloader_University(Dataset):
    def __init__(self,root,transforms,names=None):
        super(Dataloader_University).__init__()
        self.transforms_drone_street = transforms['train']
        self.transforms_satellite = transforms['satellite']
        self.root = root
        
        # 自动检测是训练集还是测试集，并设置相应的目录名称
        if 'test' in root and os.path.exists(os.path.join(root, 'query_satellite')) and os.path.exists(os.path.join(root, 'query_drone')):
            # 测试集目录结构
            self.names = ['query_satellite', 'query_street', 'query_drone', 'gallery_satellite', 'gallery_drone', 'gallery_street']
            # 对于测试，我们主要使用query_satellite和query_drone
            self.active_names = {'satellite': 'query_satellite', 'drone': 'query_drone', 'street': 'query_street'}
        else:
            # 训练集目录结构
            self.names = names if names else ['satellite','street','drone','google']
            self.active_names = {'satellite': 'satellite', 'drone': 'drone', 'street': 'street'}
        
        #获取所有图片的相对路径分别放到对应的类别中
        #{satelite:{0839:[0839.jpg],0840:[0840.jpg]}}
        dict_path = {}
        for name in self.names:
            if not os.path.exists(os.path.join(root, name)):
                continue
            dict_ = {}
            for cls_name in os.listdir(os.path.join(root, name)):
                img_list = os.listdir(os.path.join(root,name,cls_name))
                img_path_list = [os.path.join(root,name,cls_name,img) for img in img_list]
                dict_[cls_name] = img_path_list
            dict_path[name] = dict_
            # dict_path[name+"/"+cls_name] = img_path_list

        #获取设置名字与索引之间的镜像
        # 使用active_names中的第一个有效名称
        first_valid_name = None
        for key in self.active_names:
            if self.active_names[key] in dict_path:
                first_valid_name = self.active_names[key]
                break
        
        if first_valid_name is None:
            raise ValueError(f"No valid directory found in {root}")
            
        cls_names = os.listdir(os.path.join(root,first_valid_name))
        cls_names.sort()
        map_dict={i:cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2

    #从对应的类别中抽一张出来
    def sample_from_cls(self,name,cls_num):
        img_path_list = self.dict_path[name][cls_num]
        max_attempts = 5  # 最多尝试5次寻找有效样本
        attempt = 0
        
        while attempt < max_attempts:
            img_path = np.random.choice(img_path_list,1)[0]
            try:
                img = Image.open(img_path)
                # 将图像转换为RGB格式，处理RGBA或灰度图像
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 过滤图像尺寸过小的样本（宽度或高度小于100像素）
                if img.width < 100 or img.height < 100:
                    attempt += 1
                    continue
                
                # 过滤全黑或全白图像（通过检查像素值标准差）
                img_array = np.array(img)
                if np.std(img_array) < 10:  # 标准差过小，图像过于单调
                    attempt += 1
                    continue
                
                return img, img_path
            except Exception as e:
                # 捕获图像打开或处理过程中的异常
                attempt += 1
                continue
        
        # 如果多次尝试都失败，返回第一张图像作为备用
        if img_path_list:
            img_path = img_path_list[0]
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img, img_path
        else:
            raise ValueError(f"No valid images found for class {cls_num} in {name}")


    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        
        # 使用active_names中的正确目录名称
        satellite_name = self.active_names['satellite']
        img, satellite_path = self.sample_from_cls(satellite_name,cls_nums)
        img_s = self.transforms_satellite(img)

        # 尝试获取street图像，如果不存在则返回None
        img_st = None
        street_path = None
        if 'street' in self.active_names and self.active_names['street'] in self.dict_path:
            street_name = self.active_names['street']
            img, street_path = self.sample_from_cls(street_name,cls_nums)
            img_st = self.transforms_drone_street(img)

        # 使用active_names中的正确目录名称
        drone_name = self.active_names['drone']
        img, drone_path = self.sample_from_cls(drone_name,cls_nums)
        img_d = self.transforms_drone_street(img)
        return img_s,img_st,img_d,index,satellite_path,drone_path,street_path


    def __len__(self):
        return len(self.cls_names)



class Sampler_University(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8,sample_num=4):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num

    def __iter__(self):
        list = np.arange(0,self.data_len)
        np.random.shuffle(list)
        nums = np.repeat(list,self.sample_num,axis=0)
        return iter(nums)

    def __len__(self):
        return self.data_len


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    img_s,img_st,img_d,ids,satellite_paths,drone_paths,street_paths = zip(*batch)
    ids = torch.tensor(ids,dtype=torch.int64)
    return [torch.stack(img_s, dim=0),ids],[torch.stack(img_st,dim=0) if img_st[0] is not None else None,ids], [torch.stack(img_d,dim=0),ids], satellite_paths, drone_paths

if __name__ == '__main__':
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


    transform_train_list ={"satellite": transforms.Compose(transform_train_list),
                            "train":transforms.Compose(transform_train_list)}
    datasets = Dataloader_University(root="/home/dmmm/University-Release/train",transforms=transform_train_list,names=['satellite','drone'])
    samper = Sampler_University(datasets,8)
    dataloader = DataLoader(datasets,batch_size=8,num_workers=0,sampler=samper,collate_fn=train_collate_fn)
    for data_s,data_d in dataloader:
        print()


