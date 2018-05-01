#!/usr/bin/python3
# -*- coding: utf-8  -*-
import numpy
import cv2

def trainModel(images, labels):
    '''
    训练模型
    '''
    if len(images) < 2:
        raise Exception('the data set is too small');
    model = cv2.face.EigenFaceRecognizer_create();
    model.train(images, labels);
    model.save('model/facePCAModel.xml');
    print('facePCAModel training success')

    model1 = cv2.face.FisherFaceRecognizer_create();
    model1.train(images, labels);
    model1.save('model/faceFisherModel.xml');
    print('faceFisherModel training success')

    model2 = cv2.face.LBPHFaceRecognizer_create();
    model2.train(images, labels);
    model2.save('model/faceLBPHModel.xml');
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
                break;
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
    labels = numpy.array(labels)
    return (images, labels)

if __name__ == '__main__':
    # data set
    filename = 'data-set/list.txt'
    (images, labels) = loadImgs(filename)
    trainModel(images, labels)
