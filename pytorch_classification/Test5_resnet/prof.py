import os
import numpy as np
import torch
from torchvision.models import resnet18
from model import resnet34
import time


if __name__ == '__main__':
    model = resnet18(pretrained=False)
    device = torch.device('cuda')
    model = resnet34(num_classes=5)
    weights_path = "./pth_resnet/pruning2.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    model.to(device)
    dump_input = torch.ones(1,3,224,224).to(device)
    # dump_input1 = torch.ones(1,3,224,224).to(device)
    # dump_input2 = torch.ones(1,3,224,224).to(device)

    # for循环，测gpu上运行模型
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.time()
        outputs = model(dump_input)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        outputs = model(dump_input)
    print(prof.table())
    prof.export_chrome_trace('./resnet_profile.json')