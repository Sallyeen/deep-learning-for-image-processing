import torch
import torch.onnx
import netron
from model_v2 import MobileNetV2

model = MobileNetV2(num_classes=5)
input = torch.rand(16,3,224,224)
output = model(input)

onnx_path = "onnx_model_MobileNetV2.onnx"
torch.onnx.export(model, input, onnx_path)
netron.start(onnx_path)
