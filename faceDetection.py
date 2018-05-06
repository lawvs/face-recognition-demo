#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import glob

face_cascade_name = 'haarcascades/haarcascade_frontalface_default.xml'
eyes_cascade_name = 'haarcascades/haarcascade_eye_tree_eyeglasses.xml'

# 加载分类器
# 定义人脸分类器
face_cascade = cv2.CascadeClassifier(face_cascade_name)
# 定义人眼分类器
eye_cascade = cv2.CascadeClassifier(eyes_cascade_name)

def resizeImage(image, width=None, height=None, inter=cv2.INTER_AREA):
    newsize = (width, height)
    #获取图像尺寸
    (h,w) = image.shape[:2]
    if width is None and height is None:
        return image
    #高度算缩放比例
    if width is None:
        n = height / float(h)
        newsize = (int(n * w), height)
    else :
        n = width / float(w)
        newsize = (width, int(h * n))

    # 缩放图像
    newimage = cv2.resize(image, newsize, interpolation=inter)
    return newimage

def detectFaces(img):
    '''
    人脸检测
    '''
    global face_cascade
    minSize = (100, 100) # minSize 为目标的最小尺寸
    maxSize = (1000, 1000) # maxSize 为目标的最大尺寸
    faces = face_cascade.detectMultiScale(img, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, minSize, maxSize)
    return faces

def detectEyes(img):
    global eye_cascade
    eyes = eye_cascade.detectMultiScale(img, 1.1, 2, cv2.CASCADE_SCALE_IMAGE, (2, 2))
    return eyes

def saveFaces():
    '''
    保存 image 下所有人脸
    '''
    path = "image/" # 图像文件夹目录
    target = 'image/face/' # 保存位置
    files= glob.glob(path + '*.jpg')
    for file in files:
        img = cv2.imread(file)
        print('handle {}'.format(file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度
        faces = detectFaces(gray)
        for faceRect in faces:
            x, y, w, h = faceRect
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            # cv2.imshow('face' + str(x), roi_color) # face
            cv2.imwrite('{}{}-{}.jpg'.format(target, x, y), roi_color) # save

def main():
    # image
    filename = 'image/10.jpg'
    # 读取图片
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度

    # 获取图像尺寸
    height = img.shape[0]
    width = img.shape[1]
    print('image size:', width, height)

    faces = detectFaces(gray)
    print('faces num:', len(faces))

    for faceRect in faces:
        x, y, w, h = faceRect
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        # cv2.imshow('face' + str(x), roi_color) # face
        # cv2.imwrite('image/face/{}.jpg'.format(x), roi_color) # save
        # print('save image/face/{}.jpg'.format(x))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, 8, 0)

        # 人眼识别
        eyes = detectEyes(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # 缩放
    newW = 700
    scale = width / newW
    newH = int(height * scale)
    # newH = 700
    newimage = resizeImage(img, newW, newH, cv2.INTER_LINEAR)
    cv2.imshow('newimage.jpg', newimage)

    # cv2.imshow('img', resizeImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
