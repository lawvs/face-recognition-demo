#!/usr/bin/python3
# -*- coding: utf-8  -*-
import cv2

def detect(img):
    '''
    人脸检测
    '''
    # 加载分类器
    # 定义人脸分类器
    face_cascade_name = 'haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_name)

    # 获取图像尺寸
    height = img.shape[0]
    width = img.shape[1]
    print('image size:', width, height)

    # 人脸检测
    minSize = (100, 100) # minSize 为目标的最小尺寸
    maxSize = (1000, 1000) # maxSize 为目标的最大尺寸
    faces = face_cascade.detectMultiScale(img, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, minSize, maxSize)
    facesRoi = []

    for faceRect in faces:
        x, y, w, h = faceRect
        roi = img[y:y + h, x:x + w]
        # cv2.imshow('face' + str(x), roi_color) # face
        # cv2.waitKey(0)
        # cv2.imwrite('data-set/{}.jpg'.format(x), roi_color) # save
        facesRoi.append(roi)
    return facesRoi

def identify():
    '''
    识别人脸
    '''
    # image
    imgName = 'image/5.jpg'
    modelName = 'model/facePCAModel.xml'
    width = 92
    height = 112

    img = cv2.imread(imgName)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度

    faces = detect(gray)
    print('faces num:', len(faces))
    face = faces[0] if len(faces) > 0 else None

    face = cv2.resize(face, (width, height))

    faceRecognizer = cv2.face.EigenFaceRecognizer_create()
    faceRecognizer.read(modelName) # 读取模型
    print('predicting')
    label, confidence = faceRecognizer.predict(face) # 人脸识别
    print(label)
    print(confidence)

if __name__ == '__main__':
    identify()
