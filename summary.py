from model_v2 import MobileNetV2
from thop import profile
import torch

model = MobileNetV2()
input = torch.rand(16, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))

print("flops: ", flops)
print("params: ", params)