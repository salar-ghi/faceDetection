from cvzone.FaceDetectionModule import FaceDetector
import cv2
import face_recognition
import numpy as np
import pickle
import cvzone
import torch
# from numba import jit, cuda

from time import sleep
from collections import deque
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)

# print("Is CUDA supported by this system? ", {torch.cuda.is_available()})

# cuda_id = torch.cuda.current_device()
# print("Name of current CUDA device:", {torch.cuda.get_device_name(cuda_id)})



# def EncodeFiles():
#     file = open('EncodeFile.p', 'rb')
#     encodeListKnownWithIds = pickle.load(file)
#     file.close()
#     return encodeListKnownWithIds

# print('active threads 1-: ',threading.active_count())

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = 0

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
    # frm = cv2.resize(image, dim, interpolation=inter)
    
# 226
# 243

detector = FaceDetector()
global x, y, w, h , x2
rtspurl =  'rtsp://admin:ndcndc@192.168.10.226:554/channel1'
Localurl =  'rtsp://admin:admin4763@192.168.5.190:554/'
httpurl =  'http://192.168.10.226:80/video'
darourl =  'rtsp://admin:admin1234@192.168.16.252:554'


class camCapture:
    def __init__(self, camID, buffer_size):
        torch.set_default_device("cuda:0")
        self.Frame = deque(maxlen=buffer_size)
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(camID)
        
    def start(self):
        t1 = threading.Thread(target=self.queryframe, daemon=True, args=())
        t1.start()
        t2 = threading.Thread(target=self.EncodeFiles, daemon=True, args=())
        t2.start()
        # t3 = threading.Thread(target=EncodeImg , daemon=True, args=(frame))
        # t3.start()
        # t3.join()
        
    def stop(self):
        self.isstop = True

    def EncodeFiles(self):
        file = open('EncodeFile.p', 'rb')
        encodeListKnownWithIds = pickle.load(file)
        file.close()
        return encodeListKnownWithIds

    def getframe(self):
        return self.Frame.pop()

    def queryframe(self):
        while (not self.isstop):
            self.status, tmp = self.capture.read()
            self.Frame.append(tmp)

        self.capture.release()

    def StartEncodeImg(frame):
        t3 = threading.Thread(target=EncodeImg , daemon=True, args=(frame))
        t3.start()
    
    def EncodeImg(frame):
        img = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceCurFrame = face_recognition.face_locations(img)
        encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            mtchs = face_recognition.compare_faces(encodeListKnown, encodeFace)
            fceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            mtchIndx = np.argmin(fceDis)
            if fceDis[mtchIndx] < 0.5 and mtchs[mtchIndx]:
                return EmployeeIds[mtchIndx]


resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]
cam = camCapture(rtspurl, buffer_size=100)
cam.capture.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'MJPG'))
cam.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[3][0])
cam.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[3][1])
# cam.capture.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'XVID'))
cam.capture.set(cv2.CAP_PROP_BUFFERSIZE, 100)
cam.capture.set(cv2.CAP_PROP_FPS, 10)
cam.capture.set(cv2.CAP_PROP_FRAME_COUNT,1)
cam.capture.set(cv2.CAP_PROP_POS_FRAMES,1)

# cv2.waitKey(1000 / cv2.CAP_PROP_FPS - 1)

cam.start()
# sleep(2)

encodeListKnown, EmployeeIds = cam.EncodeFiles()


def EncodeImg(frame):
    img = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        mtchs = face_recognition.compare_faces(encodeListKnown, encodeFace)
        fceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        mtchIndx = np.argmin(fceDis)
        if fceDis[mtchIndx] < 0.5 and mtchs[mtchIndx]:
            return EmployeeIds[mtchIndx]
        
# def StartEncodeImg(frame):
#     t3 = threading.Thread(target=EncodeImg , daemon=True, args=(frame))
#     t3.start()
#     t3.join()

detector = FaceDetector()
while True:
    # ret, frame = cam.read()
    frame = cam.getframe()
    # frame, bboxs = threading.Thread(target=detector.findFaces, daemon=True, args=(frame))
    frame, bboxs = detector.findFaces(frame)
    for box in bboxs:
        emplyee = EncodeImg(frame)
        if emplyee is not None and emplyee is not int(0):
            x, y, w, h = box['bbox']
            x2 = x + (int(w) / 2)
            cvzone.putTextRect(frame, f'{emplyee}', (int(x2+45), y-10),2, 1, (0, 255, 25),(10, 10, 10, 0.1), cv2.BORDER_TRANSPARENT,1, 1)
            cvzone.cornerRect(frame, (x, y, w, h))
        else:
            x1, y1, w1, h1 = box['bbox']
            x3 = x1 + (int(w1) / 2)
            txt ="Unknown"
            cvzone.putTextRect(frame, f'{txt}', (int(x3+15), y1-15),2, 1, (0, 0, 255),(255, 255, 255, 0.9), cv2.BORDER_TRANSPARENT,1, 1)
            cvzone.cornerRect(frame, (x1, y1, w1, h1))
    
    frm = ResizeWithAspectRatio(frame, width=1024)
    cv2.imshow("Real-time Detection", frm)
    k = cv2.waitKey(30) & 0Xff
    if k == 27: # Press 'ESC' to quit
        cam.stop()
        break
cv2.destroyAllWindows()
        