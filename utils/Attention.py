# -*- coding: UTF-8 -*-
"""
 Time      :  2022/9/19 20:25
 File      :  CBAM.py
 Software  :  PyCharm
 Function  :  实现CBAM\CAM模块
 """

import os

import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x


# ---------------#
#     CAM函数    #
#     CoordAtt   #
# ---------------#
# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


# CBAM的主函数：
if __name__ == '__main__':
    inputs = torch.randn(10, 100, 224, 224)
    model = CBAM(in_channel=100)  # CBAM模块, 可以插入CNN及任意网络中, 输入特征图in_channel的维度
    print(model)
    outputs = model(inputs)
    print("输入维度:", inputs.shape)
    print("输出维度:", outputs.shape)
