# -*- coding: UTF-8 -*-
"""
 Time      :  2022/10/17 14:13
 File      :  inference_time.py
 Software  :  PyCharm
 Function  :  计算推理时间
 Link      ：  https://blog.csdn.net/qq_36560894/article/details/125132834?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166598674916800184179519%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166598674916800184179519&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-125132834-null-null.142^v58^control,201^v3^control_1&utm_term=%E6%8E%A8%E7%90%86%E6%97%B6%E9%97%B4&spm=1018.2226.3001.4187
"""
# # 推理时间 精确度高
# import torch
# import numpy as np
# from nets.mobilenetv3 import *
#
# model = mobilenetv3(pretrained=False)       # 15.91115633646647
# # from nets.deeplabv3_plus import DeepLab
# #
# # model = DeepLab(pretrained=False)
# device = torch.device("cuda")
# model.to(device)
# dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# repetitions = 300
# timings = np.zeros((repetitions, 1))
# # GPU-WARM-UP
# for _ in range(10):
#     _ = model(dummy_input)
# # MEASURE PERFORMANCE
# with torch.no_grad():
#     for rep in range(repetitions):
#         starter.record()
#         _ = model(dummy_input)
#         ender.record()
#         # WAIT FOR GPU SYNC
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time(ender)
#         timings[rep] = curr_time
# mean_syn = np.sum(timings) / repetitions
# std_syn = np.std(timings)
# print(mean_syn)


import torch
# from nets.ResNet50 import resnet50
# from torchvision.models.resnet import resnet101
# from nets.mobilenetv3 import *
# from nets.xception import *
# from nets.mobilenetv3_noSE import *
from nets.mobilenetv3_ECA import *
# from nets.mobilenetv2 import *
# from nets.Inceptionv3 import *
# from nets.VGG16 import *


iterations = 300   # 重复计算的轮次
# model = resnet50
# model = resnet101()                       # Inference time: 18.919903, FPS: 52.85439415259553
# model = mobilenetv3()                     # Inference time: 13.086273, FPS: 76.41595015053252 (noSE noECA)
# model = mobilenetv3()                     # Inference time: 15.973874, FPS: 62.6022212416479 (SE)
                                            # Inference time: 16.492891, FPS: 60.63218274493251
model = MobileNetV3()                     # Inference time: 14.877932, FPS: 67.2136441564392 (ECA)
                                            # Inference time: 14.579990, FPS: 68.58715084573569
                                            # Inference time: 14.255530, FPS: 70.14821440744575
# model = Xception(downsample_factor=16)    # Inference time: 22.525101, FPS: 44.39491804999983
# model = Xception(downsample_factor=8)     # Inference time: 35.231400, FPS: 28.38377166863367
# model = mobilenetv2()                     # Inference time: 8.467016, FPS: 118.10536014173869
                                            # Inference time: 7.726775, FPS: 129.42010603286639
# model = Inception3()
# model = vgg16()                            # Inference time: 14.646691, FPS: 68.27480541462982
# model = resnet50()
device = torch.device("cuda:0")
model.to(device)

random_input = torch.randn(1, 3, 224, 224).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

