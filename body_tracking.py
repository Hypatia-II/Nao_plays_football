def body_track(motionProxy, yaw):
    """
    Fonction body_track qui permet de repositionner le corps dans l'axe de la tete.

    :param motionProxy:
    :param yaw: angle de la tete
    :return: None
    """

    stiffnesses = 1.0
    motionProxy.setStiffnesses(["Body"], stiffnesses)
    motionProxy.setWalkArmsEnabled(True, True)
    motionProxy.move(0, 0, 0.3 * yaw)