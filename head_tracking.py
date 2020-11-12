
def head_track(x_ball, y_ball, integral_x, integral_y, dtLoop, motionProxy):
    w = 320
    h = 240
    (errx, erry) = (-x_ball+w/2, y_ball-h/2)
    names = ["HeadYaw", "HeadPitch"]
    yaw0, pitch0 = motionProxy.getAngles(names, True)

    integral_x = integral_x + errx*dtLoop
    integral_y = integral_y + erry*dtLoop
    Kp = 0.001
    Ki = 0

    yaw1 = yaw0 + Kp*errx + Ki*integral_x
    pitch1 = pitch0 + Kp*erry + Ki*integral_y


    angles = [yaw1, pitch1]
    fractionMaxSpeed = 0.5
    motionProxy.setAngles(names, angles, fractionMaxSpeed)
    return (yaw1, pitch1)