import numpy as np
import cv2


cap = cv2.VideoCapture(0)  # mo cam
#cap.set (4, 100)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # su dung ma nguon mo
id = 1
# name = input("Enter your name : ")
# insertOrUpdate(id, name)
sampleNum = 0
while (True):
    ret, img = cap.read()  # doc cam

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # chuyen khong gian mau BGR COLOR_BGR2GRAY

    faces = faceDetect.detectMultiScale(gray, 1.3, 5);  # nhan dang khuon mat
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("dataset/dec/" + str(id) + "." + str(sampleNum) + ".jpg", img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # ve hinh chu nhat xung quanh mat
        cv2.waitKey(100)
    cv2.imshow("face", img);
    cv2.waitKey(1)
    if (sampleNum > 80):
        break

cap.release()
cv2.destroyAllWindows()
