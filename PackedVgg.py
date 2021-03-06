# @Time : 2021-03-29 15:06
# @Author : Xuanhan Liu
# @Site : 
# @File : PackedVgg.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        # the 3rd, 8th, 15th and 22nd layer of \
        # self.features are: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)