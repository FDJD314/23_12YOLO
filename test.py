import torch

import app_net
from ultralytics import YOLO
import cv2 as cv

if __name__ == '__main__':
    model = YOLO('model/yolov8s.pt')
    source = 'source/test.jpg'
    results = model(source)
    img_r = results[0].plot()
    cv.imwrite("source/results.jpg", img_r)

    torch.save(model.model, "model/model.pt")

    # model.export()
