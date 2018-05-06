#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import faceDetection
import faceRecognition

SAVE_NAME = 'image/cap{}.jpg'

def capture():
    # camera
    cap = cv2.VideoCapture(0)
    while(True):
        # get a frame
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度
        faces = faceDetection.detectFaces(gray) # 检测人脸

        rectFrame = frame.copy()
        for faceRect in faces:
            x, y, w, h = faceRect
            roi = frame[y:y + h, x:x + w] # 人脸
            cv2.rectangle(rectFrame, (x, y), (x + w, y + h), (10, 200, 10), 2, 8, 0)

            try:
                label, confidence = faceRecognition.predict(roi) # 识别人脸
                # print(label, confidence)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(rectFrame, str(label) , (x, y - 25), font, 1.4, (200, 100, 10), thickness=2)
                cv2.putText(rectFrame, str(round(confidence)) , (x, y - 5), font, 1.4, (200, 100, 10), thickness=2)
            except Exception as e:
                print('face recognition fail')
                # raise e
        # show a frame
        cv2.imshow("capture", rectFrame)
        keycode = cv2.waitKey(1)
        if keycode & 0xFF == ord('q'):
            print('exit')
            break
        if keycode & 0xFF == ord('c'):
            imgName = SAVE_NAME.format(x)
            cv2.imwrite(imgName, frame) # save
            print('cature {}'.format(imgName))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture()
