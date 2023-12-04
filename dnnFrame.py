from cvzone.FaceDetectionModule import FaceDetector
import cv2
import torch
from time import sleep, time
from collections import deque
import threading

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)
print("Is CUDA supported by this system? ",
      {torch.cuda.is_available()})

cuda_id = torch.cuda.current_device()
print("Name of current CUDA device:",
      {torch.cuda.get_device_name(cuda_id)})



def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)



detector = FaceDetector()

rtspurl =  "rtsp://admin:ndcndc@192.168.10.226:554/channel1"
Localurl =  'rtsp://admin:admin4763@192.168.5.190:554/'
httpurl =  'http://192.168.10.226:80/video'
darourl =  'rtsp://admin:admin1234@192.168.16.252:554'


class camCapture:
    def __init__(self, camID, buffer_size):
        self.Frame = deque(maxlen=buffer_size)
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(camID)
        
    def start(self):
        t1 = threading.Thread(target=self.queryframe, daemon=True, args=())
        t1.start()

    def stop(self):
        self.isstop = True

    def getframe(self):
        return self.Frame.popleft()

    def queryframe(self):
        while (not self.isstop):
            self.status, tmp = self.capture.read()
            self.Frame.append(tmp)

        self.capture.release()

resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]
# cam = cv2.VideoCapture(Localurl, cv2.CAP_DSHOW)
cam = camCapture(0, buffer_size=50)
cam.capture.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'MJPG'))
cam.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[0][0])
cam.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[0][1])
# cam.capture.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'XVID'))
cam.capture.set(cv2.CAP_PROP_BUFFERSIZE, 100)
cam.capture.set(cv2.CAP_PROP_FPS, 10)
cam.capture.set(cv2.CAP_PROP_FRAME_COUNT,1)
cam.capture.set(cv2.CAP_PROP_POS_FRAMES,1)

# start the reading frame thread
cam.start()

# filling frames
sleep(5)

while True:
    # ret, frame = cam.read()
    frame = cam.getframe()
    frame, bboxs = detector.findFaces(frame)
    # for (i, box) in bboxs:
    #     x, y, w, h = box['bbox']
    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']        
    frm = ResizeWithAspectRatio(frame, width=960)
    cv2.imshow('Real-time Detection', frm)
    sleep( 40 / 1000) # mimic the processing time
    k = cv2.waitKey(1) & 0Xff
    if k == 27: # Press 'ESC' to quit
        cv2.destroyAllWindows()
        cam.stop()
        break
