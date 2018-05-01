#!/usr/bin/python3
# -*- coding: utf-8  -*-

import cv2
import os
import glob

def main():
    '''
    统一目录下所有jpg图像大小
    '''
    path = "" #文件夹目录
    width = 92
    height = 112

    files= glob.glob(path + '*.jpg')
    for file in files:
        img  = cv2.imread(file)
        # 获取图像尺寸
        sourceHeight = img.shape[0]
        sourceWidth = img.shape[1]
        print('Image size:', sourceWidth, sourceHeight)
        if sourceWidth == width and sourceHeight == height:
            print('process {} success(no need)'.format(file))
            continue
        newImg = cv2.resize(img, (width, height))
        cv2.imwrite(file, newImg)
        # cv2.imshow(file, newImg)
        print('process {} success'.format(file))
    cv2.waitKey(0)
    return

if __name__ == '__main__':
    main()
