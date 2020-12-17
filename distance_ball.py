import numpy as np

def params_size_ball():
    """
    Fonction params_size_ball qui permet de faire la calibration de la camera
    :return: moy: coefficient pour la calibration
    """
    db1 = np.sqrt(0.4**2+0.6**2)
    db2 = np.sqrt(0.4 ** 2 + 1 ** 2)
    lpx1 = 1.8*320/14.6
    lpx2 = 1.1*320/14.5

    coef1 = db1*lpx1/0.09
    coef2 = db2*lpx2/0.09
    moy = (coef1 +coef2) /2

    return moy
