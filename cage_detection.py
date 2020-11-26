import cv2
import imutils

def detect_cage(frame):

    redLower = (0, 100, 100)
    redUpper = (10, 255, 255)

    # resize the frame, blur it, and convert it to the HSV
    # color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "red", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask_red = cv2.inRange(hsv, redLower, redUpper)
    mask_red = cv2.erode(mask_red, None, iterations=2)
    mask_red = cv2.dilate(mask_red, None, iterations=2)

    mask_black = cv2.inRange(hsv, redLower, redUpper)
    mask_black = cv2.erode(mask_black, None, iterations=2)
    mask_black = cv2.dilate(mask_black, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts_red = imutils.grab_contours(cnts_red)

    cnts_black = cv2.findContours(mask_black.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts_black = imutils.grab_contours(cnts_black)
    center = None

    # only proceed if at least one contour was found
    if len(cnts_red) > 3:

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



    return found