import numpy as np
import cv2

def create_img_from_angle(angle, radius=16, dims=(32,32), color=(255,255,255), thickness=2):
    img = np.zeros(dims, dtype=np.uint8)
    center = (dims[0]//2, dims[1]//2)
    tip = (dims[0]//2 + int(radius * np.cos(angle)), dims[1]//2 - int(radius * np.sin(angle))) # x, y
    cv2.line(img, center, tip, color, thickness)
    return img