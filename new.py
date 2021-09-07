# 这是一个测试文件，与yolo无关，只是用来 测试一部分函数功能

import argparse
import sys

from torch import nn
from torch._C import Size
from models.common import Bottleneck, Conv
from utils.torch_utils import select_device
from utils.general import check_file
from models.yolo import Model
from PIL import Image
import os
import cv2
import numpy as np
import torch
import torchvision

from torchvision import transforms
from thop import profile

from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

cfg='models/yolov5m_mydata.yaml'
import yaml  # for torch hub
yaml_file = Path(cfg).name
with open(cfg) as f:
    yaml = yaml.safe_load(f)  # model dict
print(yaml)
# parser = argparse.ArgumentParser()
# parser.add_argument('--cfg', type=str, default='models/yolov5m_mydata.yaml', help='model.yaml')
# parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# opt = parser.parse_args()
# opt.cfg = check_file(opt.cfg) 
# device = select_device(opt.device)
# print(opt.cfg)

# model = Model(opt.cfg).to(device)
# print(model)

# a = torch.randn(size=(4, 3, 128, 128),device=torch.device("cuda"))
# y = trans(x)

# print(model(a).shape)
# class C3(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# model = C3(128)
