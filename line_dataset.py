import numpy as np
import cv2


def create_img_from_angle(angle, radius=16, dims=(32,32), color=(255,255,255), thickness=2, img=None):
    if img is None:
        img = np.zeros(dims, dtype=np.uint8)
        center = (dims[0]//2, dims[1]//2)
    else:
        center = (img.shape[0]//2, img.shape[1]//2)

    tip = (dims[0]//2 + int(radius * np.cos(angle)), dims[1]//2 - int(radius * np.sin(angle)))
    cv2.line(img, center, tip, color, thickness)
    return img


def gen_batch(batch_size, num_batches=-2, radius=16, dims=(32,32), color=(255,255,255), thickness=2):
    while num_batches == -2 or num_batches >= 0:
        num_batches -= 1
        batch = np.zeros((batch_size, *dims))
        for i in range(batch.shape[0]):
            angle = 2 * np.pi * np.random.rand()
            batch[i] = create_img_from_angle(angle, radius, color=color, thickness=thickness, img=batch[i])
        yield batch
