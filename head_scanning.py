import numpy as np

def head_scan(motionProxy, dt):
    names = ["HeadYaw", "HeadPitch"]
    yaw0, pitch0 = motionProxy.getAngles(names, True)
    fractionMaxSpeed = 0.5
    angles = [yaw0 + 0.05, 0]
    print("yaw", yaw0)
    motionProxy.setAngles(names, angles, fractionMaxSpeed)
    print("jai tourne")

    # right = True
    # if -0.1<yaw0<0.1 and right:
    #     print("ici", yaw0)
    #     angles = [yaw0 + 0.5, 0]
    #     motionProxy.setAngles(names, angles, fractionMaxSpeed)
    #     right = True
    # elif yaw0>-2 :
    #     print("la", yaw0)
    #     angles = [yaw0-0.7, 0]
    #     motionProxy.setAngles(names, angles, fractionMaxSpeed)
    #     right = False
    # else:
    #     print("coucou", yaw0)
    #     motionProxy.move(dt*np.cos(3.14/2), dt*np.sin(3.14/2), 0)