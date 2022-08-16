
import torch
from alexnet_model import AlexNet
from resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# 定义图像预处理过程，要与训练预处理过程一致
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# data_transform = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model，实例化
model = AlexNet(num_classes=5)
# model = resnet34(num_classes=5)
# load model weights
model_weight_path = "/home/gw00243982/gj/code/01-deep-learning-for-image-processing/pytorch_classification/Test2_alexnet/AlexNet.pth"  # "./resNet34.pth"，
model.load_state_dict(torch.load(model_weight_path)) # 载入模型参数
print(model)

# load image and preprocess
img = Image.open("/home/gw00243982/gj/code/data/flower_data/tulip.jpeg") 
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0) # 增加batch维度

# forward
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps,打印特征矩阵每一层的前12个通道的抽象图片
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1) # 3行，4列，每个图的索引。第一个特征矩阵的前12通道
        # [H, W, C]
        plt.imshow(im[:, :, i], cmap='gray')
    plt.show()

