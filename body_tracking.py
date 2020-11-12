def body_track(motionProxy, yaw):
    stiffnesses = 1.0
    motionProxy.setStiffnesses(["Body"], stiffnesses)
    motionProxy.setWalkArmsEnabled(True, True)
    motionProxy.move(0, 0, 0.3 * yaw)