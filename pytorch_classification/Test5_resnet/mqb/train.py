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
# from model_v2 import MobileNetV2

from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.convert_deploy import convert_deploy             # remove quant nodes for deploy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath("/home/gw00243982/gj/a02_code/data")  # get data root path
    image_path = os.path.join(data_root, "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    # 5分类网络初始化
    # net = resnet50(num_classes=5).to(device)
    net = resnet50(num_classes=5).to(device)
    
    # # 全分类网络=初始化
    # net = MobileNetV2().to(device)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./pth_mobile/mobilenet4.pth"
    model_weight_path = "../pth_resnet/resnet50.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # for param in net.parameters():
    #     param.requires_grad = False

    # # change fc layer structure

    # # mobilenet
    # net.classifier = nn.Sequential(
    #         nn.Dropout(0.2),
    #         nn.Linear(1280, 5)
    #     )
    # net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 10
    best_acc = 0
    save_path = './pth_resnet/resnet50.pth'
    train_steps = len(train_loader)
    ## --------mebench
    net.eval()
    backend = BackendType.Tensorrt
    net = prepare_by_platform(net, backend)
    enable_calibration(net)
    ## --------mebench
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    val_accurate = acc / val_num
    print('before training, val_accuracy: %.3f' %
            (val_accurate))
    
    enable_quantization(net)

    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    val_accurate = acc / val_num
    print('before training, val_accuracy: %.3f' %
            (val_accurate))

    # define dummy data for model export.
    input_shape={'data': [10, 3, 224, 224]}
    convert_deploy(net, backend, input_shape.to(device)) 
    # for epoch in range(epochs):

    #     # validate
    #     net.eval()
    #     acc = 0.0  # accumulate accurate number / epoch
    #     with torch.no_grad():
    #         val_bar = tqdm(validate_loader, file=sys.stdout)
    #         for val_data in val_bar:
    #             val_images, val_labels = val_data
    #             outputs = net(val_images.to(device))
    #             # loss = loss_function(outputs, test_labels)
    #             predict_y = torch.max(outputs, dim=1)[1]
    #             acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    #             val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
    #                                                        epochs)

    #     val_accurate = acc / val_num

    #     print('before training, [epoch %d] val_accuracy: %.3f' %
    #           (epoch + 1, val_accurate))

    #     if val_accurate > best_acc:
    #         best_acc = val_accurate
    #         torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
