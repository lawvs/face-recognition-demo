#!/usr/bin/python3
# -*- coding: utf-8  -*-
import cv2
import glob

def main():
    '''
    统一文件中所有图像大小
    '''
    listFile = 'list.txt'
    width = 92
    height = 112

    with open(listFile, 'r', encoding='utf_8') as f:
        lines = f.readlines()
        for line in lines:
            file = line.strip().split(' ')[0]
            img  = cv2.imread(file)
            # 获取图像尺寸
            sourceHeight = img.shape[0]
            sourceWidth = img.shape[1]
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
