import sys
import time
import cv2
from naoqi import ALProxy
import numpy as np
import random
import math
import os
import signal

# specific to my laptop version (do not exist on Centos Students PCs)
#import Image

try:
   from PIL import Image
except:
   import Image


def cleanKillNao(signal, frame):
    exit()


def check_constant_green_image (img,width,height):
   tstpix = []
   tstpix.append (cvImg[0,0])
   tstpix.append (cvImg[0,width-1])
   tstpix.append (cvImg[height-1,0])
   tstpix.append (cvImg[height-1,width-1])
   cstgreen = True
   for pix in tstpix:
      if pix[0] != 0:
         cstgreen = False
         break
      if (pix[1] != 154) and (pix[1] != 135):
         cstgreen = False
         break
      if pix[2] != 0:
         cstgreen = False
         break
   return cstgreen


# tiny yolo stuff
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        #print (output)
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    confidence_max = -1.0
    xok, yok, wok, hok = 0.0, 0.0, 0.0, 0.0
    idok = -1
    for i in indices:
        i = i[0]        
        confidence = confs[i]
        if confidence > confidence_max:
           confidence_max = confs[i]
           xok, yok, wok, hok = bbox[i]
           idok = classIds[i]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        st = classNames[classIds[i]].upper()+"(%.1f%%)"%(confs[i] * 100)
        cv2.putText(img, st,(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        #print (st)
        #cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        print(classIds[i],int(confs[i]*100))
    return img,xok,yok,wok,hok,idok,confidence_max 



# init tony yolo
whT = 320
confThreshold = 0.1 # detection threshold 
nmsThreshold = 0.2

classNames = ['But']
yolo_path = "/home/newubu/Robotics/nao/yolo/git/darknet/cfg"
modelConfiguration = os.path.join(yolo_path,"yolo_nao.cfg")
modelWeights = os.path.join(yolo_path,"yolov_nao.weights")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


debug = False
#debug = True

# save images for image processing setup
saveImgs = True
#saveImgs = False
imgCount = 0
maxImg = 10
camNum = 0

IP = "localhost"  # NaoQi's IP address.
PORT = 11212  # NaoQi's port
# if one NAO in the scene PORT is 11212
# if two NAOs in the scene PORT is 11212 for the first and 11216 for the second 

# Read IP address and PORT form arguments if any.
print "%3d arguments"%(len(sys.argv))
if len(sys.argv) > 1:
   IP = sys.argv[1]
if len(sys.argv) > 2:
   PORT = int(sys.argv[2])
radixImg = "img"
if len(sys.argv) > 3:
   radixImg = sys.argv[3]


# get image from vrep simulator
# set the pass to vnao to the correct path on your computer
#vnao_path = "/home/newubu/MyApps/Nao/v-rep/nao-new-model/tmp/vnao"
vnao_path = "/home/newubu/Robotics/nao/vnao/plugin-v2"
# set vnao image name 
vnao_image = "imgs/out_%5.5d.ppm"%(PORT)
cameraImage=os.path.join(vnao_path,vnao_image)

signal.signal(signal.SIGINT, cleanKillNao)

# init video
cameraProxy = ALProxy("ALVideoDevice", IP, PORT)
resolution = 1    # 0 : QQVGA, 1 : QVGA, 2 : VGA
colorSpace = 11   # RGB
#camNum = 1 # 0:top cam, 1: bottom cam
fps = 2; # 4 # frame Per Second
dtLoop = 1./fps
#cameraProxy.setParam(18, camNum)
cameraProxy.setActiveCamera(camNum)
print "Active camera is",cameraProxy.getActiveCamera()
try:
   lSubs=cameraProxy.getSubscribers()
   for subs in lSubs:
      if subs.startswith("python_client"):
         cameraProxy.unsubscribe(subs)
except:
   print "cannot unsubscribe"
   pass
try:
   videoClient = cameraProxy.subscribeCamera("python_client",camNum, 
                                       resolution, colorSpace, fps)
except:
   print "pb with subscribe"
   lSubs=cameraProxy.getSubscribers()
   for subs in lSubs:
      if subs.startswith("python_client"):
         cameraProxy.unsubscribe(subs)
   videoClient = cameraProxy.subscribeCamera("python_client",camNum,
                                       resolution, colorSpace, fps)
print cameraProxy.getSubscribers()
print "videoClient ",videoClient
# Get a camera image.
# image[6] contains the image data passed as an array of ASCII chars.
try:
   naoImage = cameraProxy.getImageRemote(videoClient)
   imageWidth = naoImage[0]
   imageHeight = naoImage[1]
   array = naoImage[6]
except:
   exit()
   
# print imageWidth,"x",imageHeight
# Create a PIL Image from our pixel array.
#pilImg = Image.fromstring("RGB", (imageWidth, imageHeight), array)
pilImg = Image.frombytes("RGB", (imageWidth, imageHeight), array)
# Convert Image to OpenCV
cvImg = np.array(pilImg)
# Convert RGB to BGR 
cvImg = cvImg[:, :, ::-1].copy()

# define display window
cv2.namedWindow("proc")
cv2.resizeWindow("proc",imageWidth,imageHeight)
cv2.moveWindow("proc",0,0)
cv2.imshow("proc",cvImg)
cv2.waitKey(1)

# if image is constant green, then we are on the simulator (no actual video frame)
cstGreen = check_constant_green_image (cvImg,imageWidth,imageHeight)
if cstGreen:
   print "run on simulated NAO, no video frame, use still images"
else:
   print "run on real NAO"

# Test getting a virtual camera image.
imgok=False
while not imgok:
   if cstGreen:
      try:
         cvImg = cv2.imread(cameraImage)
         imageHeight, imageWidth, imageChannels = cvImg.shape
         imgok=True
      except Exception, e:
         print "Can't read image %s, retry ..."%(cameraImage)
         imgok=False
         time.sleep(0.25)
   else:
      imgok=True

print "Image Size",imageWidth,imageHeight

missed = 0
#while missed < 30: 
while True: 
   t0=time.time()
   # Get current image (top cam)
   imgok=False
   found=False
   while not imgok:
      if cstGreen:
         try:
            cvImg = cv2.imread(cameraImage)
            imgok=True
         except Exception, e:
            print "Can't read image %s, retry ..."%(cameraImage)
            imgok=False
            time.sleep(0.25)
         cvImg = cv2.flip(cvImg,0)
         # Save it (just to check)
         if debug:
            cv2.imwrite ("naosimimg.png",cvImg)
      else:
         naoImage = cameraProxy.getImageRemote(videoClient)
         array = naoImage[6]
         # Create a PIL Image from our pixel array.
         pilImg = Image.frombytes("RGB", (imageWidth, imageHeight), array)
         cvImg = np.array(pilImg) # Convert Image to OpenCV
         cvImg = cvImg[:, :, ::-1].copy() # Convert RGB to BGR
         imgok=True
   if saveImgs:
      if cstGreen:
         cv2.imwrite ("/tmp/naosimu_%s_%4.4d.png"%(radixImg,imgCount),cvImg)
      else:
         cv2.imwrite ("/tmp/naoreal_%s_%4.4d.png"%(radixImg,imgCount),cvImg)
      #imgCount+=1
      #if imgCount == maxImg:
      #   break
   cv2.imshow("proc",cvImg)
   cv2.waitKey(1)

   #
   # ??? insert detection function here ???
   #
   blob = cv2.dnn.blobFromImage(cvImg, 1.0 / 255.0, (whT, whT), [0., 0., 0.], 1, crop=False)
   net.setInput(blob)
   outputNames = get_output_layers(net)
   outputs = net.forward(outputNames)
   
   dtImg,xc_obj,yc_obj,width_obj,height_obj,id_obj,confidence = findObjects(outputs, cvImg)
   print xc_obj,yc_obj,width_obj,height_obj,id_obj,confidence
   cv2.imshow("yolo",dtImg)
   cv2.waitKey(1)
   if cstGreen:
      cv2.imwrite ("/tmp/naosimu_%s_%4.4d_yolo.png"%(radixImg,imgCount),dtImg)
   else:
      cv2.imwrite ("/tmp/naoreal_%s_%4.4d_yolo.png"%(radixImg,imgCount),dtImg)
   imgCount+=1
   #if imgCount == maxImg:
   #   break
    
   #lSubs=cameraProxy.getSubscribers()
   #for subs in lSubs:
   #   if subs.startswith("python_client"):
   #      cameraProxy.unsubscribe(subs)
   #videoClient = cameraProxy.subscribeCamera("python_client",camNum,
   #                                    resolution, colorSpace, fps)

   if (found):
      missed = 0

      #
      # ??? insert head control here ???
      #
   else:
      missed += 1


      
   dt = time.time()-t0
   tSleep = dtLoop-dt
   if tSleep>0:
      time.sleep(tSleep)
   print "dtLoop = ",dtLoop,"tSleep = ",tSleep,"dt = ",dt,"frame rate = ",1./dt


sys.exit(0)
