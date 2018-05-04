#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import faceDetection

SAVE_NAME = 'image/cap{}.jpg'

def capture():
    # camera
    cap = cv2.VideoCapture(0)
    while(True):
        # get a frame
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度
        faces = faceDetection.detectFaces(gray)
        rectFrame = frame.copy()

        for faceRect in faces:
            x, y, w, h = faceRect
            # roi = frame[y:y + h, x:x + w]
            cv2.rectangle(rectFrame, (x, y), (x + w, y + h), (0, 255, 0), 2, 8, 0)

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
