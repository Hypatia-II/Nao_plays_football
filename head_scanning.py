import numpy as np

def head_scan(motionProxy):
    """
    Fonction head_scan qui permet de scanner l'environnement
    :param motionProxy:
    :return: None
    """
    names = ["HeadYaw", "HeadPitch"]
    yaw0, pitch0 = motionProxy.getAngles(names, True)
    fractionMaxSpeed = 0.5
    angles = [yaw0 + 0.05, 0]
    motionProxy.setAngles(names, angles, fractionMaxSpeed)