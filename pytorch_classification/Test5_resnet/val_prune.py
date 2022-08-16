import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34
from model import resnet50
from model_v2 import MobileNetV2
import copy
import time
from numpy import *

def main():
    # gpu setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据集加载与处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath("/home/gw00243982/gj/a02_code/data/flower_data")  # get data root path
    # image_path = os.path.join(data_root, "flower_data")  # flower data set path
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    validate_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    
    # 模型选择，权重加载
    net = resnet50(num_classes=5).to(device)
    model_weight_path = "./pth_resnet/resnet50.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    dict = torch.load(model_weight_path, map_location='cpu')
    # for name in dict:
    #     print(name, dict[name])

    # dict = torch.load(model_weight_path, map_location='cpu')
    # dict_c = copy.copy(dict)
    # for name in dict_c:
    #     if name.endswith('total_ops') or name.endswith('total_params'):
    #         del dict[name]
    # net.load_state_dict(dict)


    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    time_all = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            # 第一种计时方式--begin---
            start = time.time()
            outputs = net(val_images)
            torch.cuda.synchronize()
            end = time.time()
            time_all = time_all + (end-start)*1000
            print('Time:{}ms'.format((end-start)*1000))
            # 第一种计时方式--end---
            # # 第二种计时方式--begin---
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
            #     outputs = net(val_images.to(device))
            # print(prof.table())
            # # 第二种计时方式--end---
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    print("总时间: ", time_all)
    print("平均时间: ", time_all/23)
    val_accurate = acc / val_num
    print('val_accuracy: %.3f' %
            (val_accurate))

if __name__ == '__main__':
    main()
