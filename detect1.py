import mediapipe as mp
import cv2 
from cvzone.FaceDetectionModule import FaceDetector
import torch
import torchvision
import cvzone as cv
import asyncio
import face_recognition


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)

resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]

async def main():
    try:
        streamUrl = 'rtsp://admin:ndcndc@192.168.10.226:554/channel1'
        cam = cv2.VideoCapture(streamUrl)
        cam.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[0][0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[0][1])
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 100)
        cam.set(cv2.CAP_PROP_FPS, 10)
        cam.set(cv2.CAP_PROP_FRAME_COUNT,1)
        cam.set(cv2.CAP_PROP_POS_FRAMES,1)
        detector =  FaceDetector()
        while True:
            frame, stream = cam.read()
            if frame is None:
                print("can't open camera")
                break
            # res = Process(target=detection , args=(stream))
            # res.start()
            stream, boxes = detector.findFaces(stream)
            # print('read stream')
            for box in boxes:
                x, y, w, h = box['bbox']            
                cv.putTextRect(stream, 'Frame detected' ,(x, y - 12), 1, 1 , (0, 255,0), (50, 50, 55), cv2.FONT_ITALIC)
                cv.cornerRect(stream, (x, y, w, h))
                # yield            
            cv2.imshow('camera streaming realtime.....', stream)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                cam.release()
                break
    except:
        # print('an error occured')
         raise TypeError(Exception)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())