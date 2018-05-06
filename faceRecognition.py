#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import faceDetection

modelName = 'model/facePCAModel.xml'
faceRecognizer = cv2.face.EigenFaceRecognizer_create()
faceRecognizer.read(modelName) # 读取模型

width = 92
height = 112

def predict(face):
    '''
    识别人脸
    param face: cv2 image face only
    return: label, confidence 预测标签，可信距离
    '''
    global faceRecognizer
    face = cv2.resize(face, (width, height))
    if len(face.shape) >= 3 :
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # 灰度
    label, confidence = faceRecognizer.predict(face) # 人脸识别
    return label, confidence

def identify(img):
    '''
    识别图像中的人脸
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度

    # faces = detectRoi(img)
    faces = faceDetection.detectFaces(img)
    print('faces num:', len(faces))
    if len(faces) > 0:
        faceRect = faces[0]
        x, y, w, h = faceRect
        face = img[y:y + h, x:x + w]
    else :
        return -1, -1
    label, confidence = predict(face)
    return label, confidence

if __name__ == '__main__':
    # image
    imgName = 'image/10.jpg'

    img = cv2.imread(imgName)
    print('predicting')
    label, confidence = identify(img)
    print(label)
    print(confidence)
