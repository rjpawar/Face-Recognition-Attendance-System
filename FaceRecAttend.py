import numpy as np
import os
from datetime import datetime
import cv2
import face_recognition

img = []
label = []
dire = os.listdir('train')
# print(dire)
for lab in dire:
    trainImg = cv2.imread(f'train/{lab}')
    img.append(trainImg)
    label.append(os.path.splitext(lab)[0])
print(label)
print(len(img))

def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def getEncoding(image):
    encode = []
    for i in image:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(i)[0]
        encode.append(enc)
    return encode


encodeTrain = getEncoding(img)
print(len(encodeTrain))

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()
    frameS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
    frameLoc = face_recognition.face_locations(frameS)
    frameEncode = face_recognition.face_encodings(frameS, frameLoc)

    for encoFace, locFace in zip(frameEncode,frameLoc):
        match = face_recognition.compare_faces(encodeTrain, encoFace)
        distance = face_recognition.face_distance(encodeTrain, encoFace)
        print(distance)
        index = np.argmin(distance)

        if match[index]:
            name = label[index]
            print(name)
            y1, x2, y2, x1 = locFace
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0,0,0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Cam', frame)
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

