#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy
import cv2
import os

def trainModel(images, labels):
    '''
    训练模型
    '''
    if len(images) < 2:
        raise Exception('the data set is too small')
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(images, labels)
    model.save('model/facePCAModel.xml')
    print('facePCAModel training success')

    model1 = cv2.face.FisherFaceRecognizer_create()
    model1.train(images, labels)
    model1.save('model/faceFisherModel.xml')
    print('faceFisherModel training success')

    model2 = cv2.face.LBPHFaceRecognizer_create()
    model2.train(images, labels)
    model2.save('model/faceLBPHModel.xml')
    print('faceLBPHModel training success')
    return

def loadImgs(filename):
    '''
    加载数据
    '''
    images = []
    labels = []
    rootDir = 'data-set/'
    # 遍历文件夹
    for dir in os.listdir(rootDir):
        if not os.path.isdir(rootDir + dir):
            continue
        print('loading ' + rootDir + dir)
        # 标签为文件夹名
        label = dir
        label = int(label)
        for file in os.listdir(rootDir + dir):
            filename = rootDir + dir + '/' + file
            img = cv2.imread(filename)

            if len(img.shape) == 3 :
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度

            images.append(img)
            labels.append(label)
    return (images, labels)

def resizeImgs(images, width, height):
    '''
    统一图像大小
    '''
    list = []
    for img in images:
        # 获取图像尺寸
        sourceHeight = img.shape[0]
        sourceWidth = img.shape[1]
        if sourceWidth == width and sourceHeight == height:
            list.append(newImg)
            continue
        newImg = cv2.resize(img, (width, height))
        list.append(newImg)

        # cv2.imwrite(file, newImg)
        # cv2.imshow(file, newImg)
    return list

if __name__ == '__main__':
    # data set
    filename = 'data-set/list.txt'
    width = 92
    height = 112

    print('loading images')
    (images, labels) = loadImgs(filename)
    print('unifying image size')
    images = resizeImgs(images, width, height) # uniform size
    print('training model')
    trainModel(images, numpy.array(labels))
