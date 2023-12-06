from cvzone.FaceDetectionModule import FaceDetector
import cv2
import face_recognition
import numpy as np
import pickle
import cvzone
import torch

from time import sleep
from collections import deque
import threading
import asyncio
from multiprocessing import Process, pool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)


class camCapture:
    def __init__(self, camID, buffer_size):
        torch.set_default_device("cuda:0")
        self.Frame = deque(maxlen=buffer_size)
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(camID)
        
    def start(self):
        t1 = threading.Thread(target=self.queryframe,daemon=True ,args=())
        t1.start()
        
    def stop(self):
        self.isstop = True

    def getframe(self):
        return self.Frame.pop()

    def queryframe(self):
        while (not self.isstop):
            self.status, tmp = self.capture.read()
            self.Frame.append(tmp)

        self.capture.release()

    # def StartEncodeImg(frame):
    #     t3 = threading.Thread(target=EncodeImg , daemon=True, args=(frame))
    #     t3.start()
    
#################### load dataset ##########################
class EncodeDataset:
    def __init__(self):
        torch.set_default_device("cuda:0")
    
    def LoadFile(self):
        p1 = threading.Thread(target=self.EncodeFiles, daemon=True, args=())
        p1.start

    def EncodeFiles(self):
        file = open('EncodeFile.p', 'rb')
        encodeListKnownWithIds = pickle.load(file)
        file.close()
        return encodeListKnownWithIds

## **
dataFile = EncodeDataset()
dataFile.LoadFile()
encodeListKnown, EmployeeIds = dataFile.EncodeFiles()
    
#################### encode images ##########################
# class EncodeImages:
#     def __init__(self, frame):
#         torch.set_default_device("cuda:0")
#         self.Frame = frame
    
#     async def EncodeImg(self):
#         img = cv2.resize(self.Frame, (0, 0), None, 0.25, 0.25)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faceCurFrame = await face_recognition.face_locations(img)
#         encodeCurFrame = await face_recognition.face_encodings(img, faceCurFrame)
#         for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
#             mtchs = await face_recognition.compare_faces(encodeListKnown, encodeFace)
#             fceDis = await face_recognition.face_distance(encodeListKnown, encodeFace)
#             mtchIndx = np.argmin(fceDis)
#             if fceDis[mtchIndx] < 0.5 and mtchs[mtchIndx]:
#                 return await EmployeeIds[mtchIndx]
            
#     def StartEncodeImg(self):
#         p1 = threading.Thread(target=self.EncodeImg ,daemon=True, args=())
#         p1.start()

## **
# encode = EncodeImages()
# encode.StartEncodeImg()


#################### encode images ##########################
class ResizeWindow():
    def __init__(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        torch.set_default_device("cuda:0")        
        self.image = image
        self.width = width
        self.height = height
        self.inter = inter

    # def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    def ResizeWithAspectRatio(self):
        dim = None
        (h, w) = self.image.shape[:2]
        r = 0

        if self.width is None and self.height is None:
            return self.image
        if self.width is None:
            r = self.height / float(h)
            dim = (int(w * r), self.height)
        else:
            r = self.width / float(w)
            dim = (self.width, int(h * r))

        return cv2.resize(self.image, dim, interpolation=self.inter)

    def Resizing(self):
        t1 = threading.Thread(target=self.ResizeWithAspectRatio, daemon=True, args=() )
        t1.start()

## **
# resize = ResizeWindow()
# resize.Resizing()


#################### box detector ##########################



# encodeListKnown, EmployeeIds = cam.EncodeFiles()


class faceDetection():
    def __init__(self, Frame):
        self.frame = Frame

    async def detector(self):
        detector = FaceDetector()
        ret, bboxs = detector.findFaces(self.frame)
        for box in bboxs:
            emplyee = asyncio.run_coroutine_threadsafe(self.EncodeImg())
            print('start to get employee')
            print(emplyee)
            if emplyee is not None and emplyee is not int(0):
                x, y, w, h = box['bbox']
                x2 = x + (int(w) / 2)
                return x, y, w, h ,x2, ret, emplyee

    def detectionStart(self):
        p1 = Process(target=self.detector, args=())
        p1.start()
        p1.join()
        t1 = threading.Thread(target=asyncio.run_coroutine_threadsafe(self.EncodeImg) ,daemon=True, args=())
        t1.start()

    async def EncodeImg(self):
        img = cv2.resize(self.Frame, (0, 0), None, 0.25, 0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceCurFrame = asyncio.run(face_recognition.face_locations(img))
        encodeCurFrame = asyncio.run(face_recognition.face_encodings(img, faceCurFrame))
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            mtchs = asyncio.run(face_recognition.compare_faces(encodeListKnown, encodeFace))
            fceDis = asyncio.run(face_recognition.face_distance(encodeListKnown, encodeFace))
            mtchIndx = np.argmin(fceDis)
            if fceDis[mtchIndx] < 0.5 and mtchs[mtchIndx]:
                return await EmployeeIds[mtchIndx]

# global x, y, w, h , x2
rtspurl =  'rtsp://admin:ndcndc@192.168.10.226:554/channel1'
Localurl =  'rtsp://admin:admin4763@192.168.5.190:554/'
httpurl =  'http://192.168.10.226:80/video'
darourl =  'rtsp://admin:admin1234@192.168.16.252:554'

async def main():
    try:
        resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]
        cam = camCapture(rtspurl, buffer_size=10000)
        cam.capture.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'MJPG'))
        cam.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[3][0])
        cam.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[3][1])
        cam.capture.set(cv2.CAP_PROP_BUFFERSIZE, 100)
        cam.capture.set(cv2.CAP_PROP_FPS, 10)
        cam.capture.set(cv2.CAP_PROP_FRAME_COUNT,1)
        cam.capture.set(cv2.CAP_PROP_POS_FRAMES,1)
        
        cam.start()
        await asyncio.sleep(1)

        while True:
            frame = cam.getframe()
            # frame, bboxs = detector.findFaces(frame)
            # encode = EncodeImages(frame)
            # encode.StartEncodeImg()
            detector = faceDetection(frame)
            detector.detectionStart()
            print('it comes here !*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!')
            # for i in await detector.detector():
            x, y, w, h ,x2, ret, emplyee = asyncio.run_coroutine_threadsafe(detector.detector())
            if emplyee is not None:
                cvzone.putTextRect(frame, f'{emplyee}', (int(x2+45), y-10),2, 1, (0, 255, 25),(10, 10, 10, 0.1), cv2.BORDER_TRANSPARENT,1, 1)
                cvzone.cornerRect(frame, (x, y, w, h))
            else: 
                txt ="Unknown"
                cvzone.putTextRect(frame, f'{txt}', (int(x2+15), y-15),2, 1, (0, 0, 255),(255, 255, 255, 0.9), cv2.BORDER_TRANSPARENT,1, 1)
                cvzone.cornerRect(frame, (x, y, w, h))
            # for box in bboxs:
            #     emplyee = await encode.EncodeImg()
            #     if emplyee is not None and emplyee is not int(0):
            #         x, y, w, h = box['bbox']
            #         x2 = x + (int(w) / 2)
            #         cvzone.putTextRect(frame, f'{emplyee}', (int(x2+45), y-10),2, 1, (0, 255, 25),(10, 10, 10, 0.1), cv2.BORDER_TRANSPARENT,1, 1)
            #         cvzone.cornerRect(frame, (x, y, w, h))
            #     else:
            #         x1, y1, w1, h1 = box['bbox']
            #         x3 = x1 + (int(w1) / 2)
            #         txt ="Unknown"
            #         cvzone.putTextRect(frame, f'{txt}', (int(x3+15), y1-15),2, 1, (0, 0, 255),(255, 255, 255, 0.9), cv2.BORDER_TRANSPARENT,1, 1)
            #         cvzone.cornerRect(frame, (x1, y1, w1, h1))
            resize = ResizeWindow(frame, width=1024)
            resize.Resizing()
            frm = resize.ResizeWithAspectRatio()
            cv2.imshow("Real-time Detection", frm)
            k = cv2.waitKey(30) & 0Xff
            if k == 27: # Press 'ESC' to quit
                cam.stop()
                break
    except Exception as x:
        print('\n')
        print('error occured')
    finally:
        cv2.destroyAllWindows()


if __name__== '__main__':
    asyncio.run(main())