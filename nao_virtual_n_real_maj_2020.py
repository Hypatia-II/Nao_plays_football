# On realise les imports
import sys
import time
import cv2
from naoqi import ALProxy
import numpy as np
import os
import signal
from ball_tracking import ball_tracking
from head_tracking import head_track
from head_scanning import head_scan
from body_tracking import body_track
from distance_ball import params_size_ball
from cage_detection import get_output_layers, detect_cage
#import Image
try:
   from PIL import Image
except:
   import Image


def cleanKillNao(signal, frame):
   global postureProxy,motionProxy
   print "pgm interrupted, put NAO is safe pose ..."
   postureProxy.goToPosture("Crouch", 0.5)
   time.sleep(0.5)
   stiffnesses  = 0.0
   motionProxy.setStiffnesses(["Body"], stiffnesses)
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


debug=False
#debug=True

# save images for image processing setup
#saveImgs=True
saveImgs=False
imgCount=0

# IP = "localhost"  # NaoQi's IP address.
# PORT = 11212  # NaoQi's port
# if one NAO in the scene PORT is 11212
# if two NAOs in the scene PORT is 11212 for the first and 11216 for the second
IP = "172.20.25.50"  # NaoQi's IP address.
PORT = 9559  # NaoQi's port

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
vnao_path = "/home/ines/Documents/ENSTA/3A/Asservissement_visuel/Benito/UE52-VS-IK/"
# set vnao image name 
vnao_image = "imgs/out_%5.5d.ppm"%(PORT)
cameraImage=os.path.join(vnao_path,vnao_image)

# init motion
try:
   motionProxy = ALProxy("ALMotion", IP, PORT)
except Exception, e:
   print "Could not create proxy to ALMotion"
   print "Error was: ", e
   exit(1)

# init posture
try:
   postureProxy = ALProxy("ALRobotPosture", IP, PORT)
except Exception, e:
   print "Could not create proxy to ALPosture"
   print "Error was: ", e
   exit(1)


# work ! set current to servos
stiffnesses  = 1.0
motionProxy.wakeUp()
postureProxy.goToPosture("Crouch", 0.5)
#time.sleep(0.5)

# relax all servos by removing current (prevent over heating)
stiffnesses  = 0.0
motionProxy.setStiffnesses(["Body"], stiffnesses)
#time.sleep(0.5)

names  = ["HeadYaw", "HeadPitch"]
stiffnesses  = 1.0   # only activate head pitch and yaw servos
motionProxy.setStiffnesses(names, stiffnesses)
angles  = [0.0, 0.0]
fractionMaxSpeed  = 1.0
motionProxy.setAngles(names, angles, fractionMaxSpeed)

signal.signal(signal.SIGINT, cleanKillNao)

# init video
cameraProxy = ALProxy("ALVideoDevice", IP, PORT)
resolution = 1    # 0 : QQVGA, 1 : QVGA, 2 : VGA
colorSpace = 11   # RGB
camNum = 0 # 0:top cam, 1: bottom cam
fps = 4 # frame Per Second
dtLoop = 1./fps
cameraProxy.setParam(18, camNum)
integral_x = 0
integral_y = 0
cage = 0
cage_found = 0
params_ball = params_size_ball()

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
naoImage = cameraProxy.getImageRemote(videoClient)
imageWidth = naoImage[0]
imageHeight = naoImage[1]
array = naoImage[6]
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
# cv2.imshow("proc",cvImg)
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


# init tiny yolo
yolo_path = "/home/ines/Documents/ENSTA/3A/Asservissement_visuel/Benito/UE52-VS-IK/benito_theobald"
modelConfiguration = os.path.join(yolo_path, "yolo_nao.cfg")
modelWeights = os.path.join(yolo_path, "yolov_nao.weights")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
whT = 320
confThreshold = 0.65  # detection threshold
nmsThreshold = 0.2
classNames = ['But']

blob = cv2.dnn.blobFromImage(cvImg, 1.0 / 255.0, (whT, whT), [0., 0., 0.], 1, crop=False)
net.setInput(blob)
outputNames = get_output_layers(net)
outputs = net.forward(outputNames)

start_scan = True


print "Image Size",imageWidth,imageHeight


missed = 0
motionProxy.wakeUp()
while missed < 120:
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
         cv2.imwrite ("naosimu_%4.4d.png"%(imgCount),cvImg)
      else:
         cv2.imwrite ("naoreal_%4.4d.png"%(imgCount),cvImg)
      imgCount+=1
   # cv2.imshow("proc",cvImg)
   cv2.waitKey(1)

   # Appel de le fonction ball_tracking pour detecter les balles jaunes
   found = ball_tracking(cvImg)

   # Si une balle a ete trouvee, on recupere les coordonnees x et y de cette derniere et on met la variable found a True
   if found !=0:
      x_ball = found[0]
      y_ball = found[1]
      radius_ball = found[2]
      found = 1

   #Si on a trouve une balle ou si on cherche des cages apres avoir trouve la balle
   if (found or cage):

      # Si on regarde par la camera frontale et que l'on ne cherche pas les cages, on appelle la fonction head_track
      # pour aligner la tete avec la balle. Puis on aligne le corps avec la tete en faisant appel a la fonction body_track
      # si l'angle de la tete est superieur a 0.05rad. Quand le corps est aligne, il avance vers la balle et si l'angle
      # d'inclinaison de la tete depasse 0.6rad (ce qui equivaut a une balle proche) on passe a la camera bucale et on
      # releve la tete
      if (camNum == 0) and (cage == 0):
         missed = 0
         yaw, pitch = head_track(x_ball, y_ball, integral_x, integral_y, dtLoop, motionProxy)
         if (abs(yaw)>0.05):
            body_track(motionProxy, yaw)
         else :
            distance = 0.09*params_ball/(2*radius_ball)
            # motionProxy.move(0.2*distance/(dtLoop), 0, 0)
            motionProxy.move(0.09 * distance / (dtLoop), 0, 0)
            if pitch > 0.6:
               camNum = 1
               try:
                  videoClient = cameraProxy.subscribeCamera("python_client", camNum, resolution, colorSpace, fps)
               except:
                  print "pb with subscribe"
                  lSubs = cameraProxy.getSubscribers()
                  for subs in lSubs:
                     if subs.startswith("python_client"):
                        cameraProxy.unsubscribe(subs)
                  videoClient = cameraProxy.subscribeCamera("python_client", camNum, resolution, colorSpace, fps)

               motionProxy.stopMove()
               angles = [0, 0.4]
               fractionMaxSpeed = 0.5
               motionProxy.setAngles(names, angles, fractionMaxSpeed)

      # Une fois la camera bucale activee et si l'on ne cherche toujours pas les cages, on positionne le pied droit du
      # robot en face de la balle. On passe donc a la camera frontale et on se met a chercher la balle en passant cage
      # a True

      elif (camNum == 1) and (cage == 0):

         x_pied = 205
         y_pied = 174
         distance_x = x_pied - x_ball
         distance_y = y_pied - y_ball
         # Pour faire avancer le robot en x mettre la coord en y dans moveTo
         if (abs(distance_x) > 15):
            motionProxy.moveTo(0,10**(-3)*distance_x,0)
         else:
            motionProxy.stopMove()
            cage = 1
            camNum = 0
            try:
               videoClient = cameraProxy.subscribeCamera("python_client", camNum, resolution, colorSpace, fps)
            except:
               print "pb with subscribe"
               lSubs = cameraProxy.getSubscribers()
               for subs in lSubs:
                  if subs.startswith("python_client"):
                     cameraProxy.unsubscribe(subs)
               videoClient = cameraProxy.subscribeCamera("python_client", camNum, resolution, colorSpace, fps)

      # Lorsque l'on cherche les cages et que l'on utilise la camera frontale, on detecte la presence de cages avec
      # l'algorithme de machine learning de la fonction detect_cage. Si aucune cage n'a ete trouvee, on demarre un scan
      # de l'environnement par la droite. Lorsque la  cage est trouvee, on tourne en arc de cercle autour de la balle
      # pour se placer en face des cages. On ne cherche donc plus les cages, cage est passe a False.
      elif (camNum == 0) and (cage == 1):

         blob = cv2.dnn.blobFromImage(cvImg, 1.0 / 255.0, (whT, whT), [0., 0., 0.], 1, crop=False)
         net.setInput(blob)
         outputNames = get_output_layers(net)
         outputs = net.forward(outputNames)
         cage_found, dtImg= detect_cage(outputs, cvImg,confThreshold,nmsThreshold,classNames)
         print("cage trouvee", cage_found)

         if cage_found == 0:
            # turn head
            # test scan
            names = ["HeadYaw", "HeadPitch"]
            yaw0, pitch0 = motionProxy.getAngles(names, True)
            fractionMaxSpeed = 0.8
            angles = [-2.07, 0]
            if start_scan:
               motionProxy.setAngles(names, angles, fractionMaxSpeed)
               time.sleep(1)
               start_scan = False
            else:
               head_scan(motionProxy)

         elif cage_found == 1:
            confThreshold = 0.3
            yaw0, pitch0 = motionProxy.getAngles(names, True)
            print("je suis la")
            if (abs(yaw0) > 0.2):
               print("j'ajuste position")
               sign = yaw0 / abs(yaw0)
               # theta = abs(yaw0)
               print("yaw", yaw0)
               theta = 1.5
               # motionProxy.moveTo( 0, 3 * sign * np.cos(theta), -1.5 * yaw0)
               motionProxy.move(0, -0.5 * sign * np.cos(theta), 0.1*theta* yaw0)
               print("j'aligne ma tete")
               names = ["HeadYaw", "HeadPitch"]
               fractionMaxSpeed = 0.3
               #angles = [yaw0 - 0.08*theta*yaw0, 0]
               angles = [yaw0 - 0.03 * theta * yaw0, 0]
               motionProxy.setAngles(names, angles, fractionMaxSpeed)
            else:
               print("je shoote")
               motionProxy.moveTo(30, 0, 0)
         cv2.imshow("yolo", dtImg)
         cv2.waitKey(1)

         names = ["HeadYaw", "HeadPitch"]
         yaw0, pitch0 = motionProxy.getAngles(names, True)

   else:
      missed += 1
   dt = time.time()-t0
   tSleep = dtLoop-dt
   if tSleep>0:
      time.sleep(tSleep)
   print "dtLoop = ",dtLoop,"tSleep = ",tSleep,"dt = ",dt,"frame rate = ",1./dt


# relax !  no current in servos
print postureProxy.getPostureList()
postureProxy.goToPosture("Crouch", 0.5)
stiffnesses = 0.0
motionProxy.setStiffnesses(["Body"], stiffnesses)

sys.exit(0)
