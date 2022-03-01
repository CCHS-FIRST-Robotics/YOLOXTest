import numpy as np
import cv2
import YoloxDeploy

y = YoloxDeploy.YoloxDeploy()
cam = cv2.VideoCapture(1)

while (True):
    _, frame = cam.read()
    y.deploy(frame)