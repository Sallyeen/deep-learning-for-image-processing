from model import resnet34 # 导入模型
from thop import profile # 导入工具
import torch

model = resnet34(num_classes=5) # 模型初始化，看看是否需要改分类数
input = torch.rand(1, 3, 224, 224) # 输入的随机初始化
flops, params = profile(model, inputs=(input, )) # 工具


print("flops: ", flops)
print("params: ", params)