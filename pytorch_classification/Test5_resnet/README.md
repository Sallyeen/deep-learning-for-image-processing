## 文件结构：
```
  ├── model.py: ResNet模型搭建
  ├── train.py: 训练脚本
  ├── predict.py: 单张图像预测脚本
  └── batch_predict.py: 批量图像预测脚本
```
```
.
├── main.py         # 剪枝功能文件，导入模型，可调用剪枝函数进行剪枝，输出剪枝后的pth文件（模型参数）
├── model.py        # 模型文件
├── pruning0.pth    # 实验二所得的pth模型参数文件
├── pruning1.pth    # 实验三所得的pth模型参数文件
├── pruning2.pth    # 实验四所得的pth模型参数文件
├── resnet34.pth    # 实验一（正常训练）所得的pth模型参数文件
├── train.py        # 正常训练脚本
└── val_prune.py    # 精度验证脚本
```