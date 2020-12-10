import cv2
import numpy as np

try:
   from PIL import Image
except:
   import Image


# tiny yolo stuff
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def detect_cage(outputs,img, confThreshold, nmsThreshold, classNames):
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
    if len(indices)!=0:
        found = 1
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
    else:
        found = 0

    return found, img,xok,yok,wok,hok,idok,confidence
