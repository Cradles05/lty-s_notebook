# -*- coding: UTF-8 -*-
"""
 Time      :  2022/10/13 11:08
 File      :  change_file_name.py
 Software  :  PyCharm
 Function  :  删除图片名称中的空格
"""
import os.path


def rename(img_folder, num):
    for img_name in os.listdir(img_folder):  # os.listdir()： 列出路径下所有的文件
        print(img_name + '变为', end='')
        # os.path.join() 拼接文件路径
        src = os.path.join(img_folder, img_name)  # src：要修改的目录名
        # dst = os.path.join(img_folder, img_name.split('t')[0] + 'tCute' + str(
        #     num) + '.jpg')  # dst： 修改后的目录名      注意此处str(num)将num转化为字符串,继而拼接
        dst = os.path.join(img_folder, img_name.replace(" ", "").split('\n')[0])
        # num = num + 1
        os.rename(src, dst)  # 用dst替代src
        print(dst)


def main():
    img_folder0 = 'C:\\Users\\PC\\Documents\\Python\\deeplabv3-plus-pytorch\\datasets\\VOCdevkit\\VOC2007' \
                  '\\SegmentationClass '
    # JPEGImages SegmentationClass图片的文件夹路径    直接放你的文件夹路径即可
    num = 1
    rename(img_folder0, num)


if __name__ == "__main__":
    main()


# from pathlib2 import Path
#
# # 图片存放路径
# imagePath = 'E:\\stuphoto\\imageTest\\'
# imageSize = 0
# # 批量去除图片名空格
# if __name__ == '__main__':
#     # "*"代表所有图片，可用于筛选
#     for item in Path(imagePath).rglob("*"):
#         # 获取图片名
#         # replace()遍历字符串 --》替换
#         imgName = item.name.replace(" ", "")
#         # 重命名图片名  路径加图片名
#         item.rename(imagePath + imgName)
#         print(str(imageSize) + " 张图片"" 去空格 成功: " + imgName)
#         imageSize = imageSize + 1
#
