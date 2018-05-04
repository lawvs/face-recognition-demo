#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy
import cv2

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

    with open(filename, 'r', encoding='utf_8') as f:
        lines = f.readlines()
        for line in lines:
            if line == ' ':
                break
            sample = line.strip().split(' ')
            if len(sample) < 2:
                continue
            img = cv2.imread(sample[0])

            label = sample[1]
            label = int(label)

            if len(img.shape) == 3 :
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度

            images.append(img)
            labels.append(label)
    return (images, labels)

def resizeImgs(images, width, height):
    '''
    统一图像大小
    '''
    for img in images:
        # 获取图像尺寸
        sourceHeight = img.shape[0]
        sourceWidth = img.shape[1]
        if sourceWidth == width and sourceHeight == height:
            continue
        newImg = cv2.resize(img, (width, height))
        cv2.imwrite(file, newImg)
        # cv2.imshow(file, newImg)
        print('process {} success'.format(file))

if __name__ == '__main__':
    # data set
    filename = 'data-set/list.txt'
    width = 92
    height = 112

    print('loading images')
    (images, labels) = loadImgs(filename)
    print('unifying image size')
    resizeImgs(images, width, height) # uniform size
    print('training model')
    trainModel(images, numpy.array(labels))
