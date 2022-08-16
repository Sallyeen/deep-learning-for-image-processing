from torchstat import stat
from model import resnet34
from model_v2 import MobileNetV2


# resnet_18 = torchvision.models.resnet18()
# resnet_34 = torchvision.models.resnet34()
model = resnet34(num_classes=5)
# eff = EfficientNet.from_pretrained('efficientnet-b0')
# model = MobileNetV2(num_classes=5)
stat(model, (3,224,224))
