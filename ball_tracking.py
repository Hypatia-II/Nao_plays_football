# import the necessary packages
import cv2
import imutils

def ball_tracking(frame):
    """
    Fonction qui analyse l'image pour detecter une balle jaune
    :param frame: Image
    :return: found
                - found = x_ball, y_ball, radius_ball
                - found = 0
    """
    yellowLower = (20, 100, 100)
    yellowUpper = (30, 255, 255)

    # resize the frame, blur it, and convert it to the HSV
    # color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "yellow", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:

        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        found = int(x), int(y), radius

    else:
        found = 0

    # show the frame to our screen

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    return (found)