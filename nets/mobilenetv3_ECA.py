# -*- coding: UTF-8 -*-
"""
 Time      :  2022/10/17 14:28
 File      :  mobilenet3_ECA.py
 Software  :  PyCharm
 Function  :  SE替换为ECA
"""
import math
import torch
import torch.nn as nn
import torchvision


BatchNorm2d = nn.BatchNorm2d

# hardswish激活函数
class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6


# 卷积+BN+ReLU
def ConvBNActivation(in_channels, out_channels, kernel_size, stride, activate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=(kernel_size - 1) // 2, groups=in_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish()
    )


# 1x1卷积+BN+ReLU
def Conv1x1BNActivation(in_channels, out_channels, activate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish()
    )


# 1x1卷积，用于调整通道数
def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels)
    )


# ----------------------------#
#   SE block换为eca_block
# ----------------------------#
class SqueezeAndExcite(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(SqueezeAndExcite, self).__init__()
        se_kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        se_kernel_size = se_kernel_size if se_kernel_size % 2 else se_kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=se_kernel_size, padding=(se_kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
        # 扩展张量中某维数据的尺寸，它括号内的输入参数是另一个张量，作用是将输入tensor的维度扩展为与指定tensor相同的size。


# ------------------------#
#   SE block Bottleneck
# ------------------------#


class SEInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride,
                 activate, use_se, se_kernel_size=1):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        # mid_channels = (in_channels * expansion_factor)

        self.conv = Conv1x1BNActivation(in_channels, mid_channels, activate)
        self.depth_conv = ConvBNActivation(mid_channels, mid_channels, kernel_size, stride, activate)
        if self.use_se:
            self.SEblock = SqueezeAndExcite(channel=mid_channels)

        self.point_conv = Conv1x1BNActivation(mid_channels, out_channels, activate)

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.depth_conv(self.conv(x))
        if self.use_se:
            out = self.SEblock(out)
        out = self.point_conv(out)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        return out


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, type='large'):
        super(MobileNetV3, self).__init__()
        self.type = type

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),
        )
        self.features = self.first_conv                 # 1 (512,512,3)->(256,256,16)
        # self.low_featrue_layer = nn.Sequential(
        #     SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1,
        #                          activate='relu', use_se=False),
        #     SEInvertedBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=2,
        #                          activate='relu', use_se=False)
        # )  # low_level_channels
        interverted_setting = [
            # in_channels,mid_channels,out_channels,kernel_size,stride,activate,use_se,se_kernel_size
            [16, 16, 16, 3, 1, 'relu', False, 0],       # 2 (256,256,16)->(256,256,16)
            [16, 64, 24, 3, 2, 'relu', False, 0],       # 3 (256,256,16)->(128,128,24)
            [24, 72, 24, 3, 1, 'relu', False, 0],       # 4 (128,128,24)->(,,40)
            [24, 72, 40, 5, 2, 'relu', True, 28],       # 5 (128,128,40)->(64,64,40)
            [40, 120, 40, 5, 1, 'relu', True, 28],
            [40, 120, 40, 5, 1, 'relu', True, 28],
            [40, 240, 80, 3, 1, 'hswish', False, 0],
            [80, 200, 80, 3, 1, 'hswish', False, 0],
            [80, 184, 80, 3, 2, 'hswish', False, 0],    # 10 (64,64,80)->(32,32,80)
            [80, 184, 80, 3, 1, 'hswish', False, 0],
            [80, 480, 112, 3, 1, 'hswish', True, 14],
            [112, 672, 112, 3, 1, 'hswish', True, 14],
            [112, 672, 160, 5, 2, 'hswish', True, 7],   # 14(32,32,112)->(16,16,160)
            [160, 960, 160, 5, 1, 'hswish', True, 7],
            [160, 960, 160, 5, 1, 'hswish', True, 7]    # 16(,,160)->(16,16,160)
        ]
        for in_channels, mid_channels, out_channels, kernel_size, stride, activate, use_se, se_kernel_size in interverted_setting:
            self.features.append(SEInvertedBottleneck(in_channels, mid_channels, out_channels, kernel_size, stride, activate, use_se, se_kernel_size))

        self.large_stage = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=1280, kernel_size=1, stride=1),
            nn.BatchNorm2d(1280),
            HardSwish(inplace=True)
        )
        self.large_last_stage = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Conv2d(in_channels=1280, out_channels=1280, kernel_size=1, stride=1),
            HardSwish(inplace=True),
        )
        self.features.append(self.large_stage)
        self.features.append(self.large_last_stage)
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(in_features=1280, out_features=num_classes)
        self._initialize_weights()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:5](x)
        x = self.features[5:](low_level_features)
        return low_level_features, x
    # def forward(self, x):
    #     x = self.features(x)
    #     # x = self.first_conv(x)  # torch.Size([2, 16, 256, 256])
    #     # if self.type == 'large':
    #     #     low_featrue_layer = self.low_featrue_layer(x)  # torch.Size([2, 24, 128, 128])
    #     #     x = self.large_bottleneck(x)  # torch.Size([2, 160, 16, 16])
    #     #     x = self.large_last_stage(x)  # torch.Size([2, 1280, 10, 10])
    #     # else:
    #     #     x = self.small_bottleneck(x)
    #     #     x = self.small_last_stage(x)
    #     # x = self.classifier(x)
    #     x = x.view(x.size(0), -1)  # torch.Size([2, 128000])
    #     # x = x.view(batchsize, -1)中batchsize指转换后有几行，
    #     # 而-1指在不告诉函数有多少列的情况下，
    #     # 根据原tensor数据和batchsize自动分配列数。
    #     out = self.classifier(x)
    #     return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = MobileNetV3(type='large')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
