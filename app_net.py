from typing import Union
from pathlib import Path
import threading
import cv2 as cv
import numpy as np
import torch
from ultralytics.utils import ops


class MyNet:
    def __init__(self, modelfile: Union[str, Path]):
        self.device = 'cpu'
        self._lock = threading.Lock()
        self.imgsz = [640, 640]

        self.model = torch.load(modelfile)

        self.model.eval()
        # self.warmup()

    def __call__(self, data=None):
        path = data["path"]
        if data["mode"] == "image":
            img_r = self.run(cv.imread(path))
            cv.imwrite("tmp/results.jpg", img_r)
        elif data["mode"] == "video":
            cap = cv.VideoCapture(data["path"])
            fourcc, fps = cap.get(cv.CAP_PROP_FOURCC), cap.get(cv.CAP_PROP_FPS)
            h, w = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            fourcc = cv.VideoWriter.fourcc(*"mp4v")
            video = cv.VideoWriter("tmp/results.mp4", fourcc, fps, (w, h))

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                img_r = self.run(frame)
                video.write(img_r)

            cap.release()
            video.release()

    # def warmup(self, imgsz=(1, 3, 640, 640)):
    #     im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
    #     for _ in range(2 if self.jit else 1):
    #         self.forward(im)  # warmup

    def run(self, im0s):
        with self._lock:
            im = self.preprocess(im0s)
            preds = self.model(im)
            im_r = self.postprocess(preds, im, im0s)
        return im_r

    def resize(self, img):
        c_shape = img.shape[:2]
        r = min(self.imgsz[0] / c_shape[0], self.imgsz[1] / c_shape[1])

        new_unpad = int(round(c_shape[1] * r)), int(round(c_shape[0] * r))
        dw, dh = self.imgsz[0] - new_unpad[0], self.imgsz[1] - new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, 32) / 2, np.mod(dh, 32) / 2  # wh padding

        if c_shape[::-1] != new_unpad:  # resize
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT,
                                value=(114, 114, 114))  # add border
        return img

    @staticmethod
    def scale_boxes(img1_shape, boxes, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding

        boxes[[0, 2]] -= pad[0]  # x padding
        boxes[[1, 3]] -= pad[1]  # y padding
        boxes[:4] /= gain

        boxes[0] = boxes[0].clamp(0, img0_shape[1])  # x1
        boxes[1] = boxes[1].clamp(0, img0_shape[0])  # y1
        boxes[2] = boxes[2].clamp(0, img0_shape[1])  # x2
        boxes[3] = boxes[3].clamp(0, img0_shape[0])  # y2
        return boxes

    def preprocess(self, im):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = self.resize(im)
            cv.imwrite('tmp.jpg', im)
            im = im.transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.float()
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return torch.unsqueeze(im, dim=0)

    def postprocess(self, preds, img, orig_img):
        pred = ops.non_max_suppression(preds, 0.6, 0.7, agnostic=False, max_det=300, classes=None)[0]

        # if not isinstance(orig_img, list):  # input images are a torch.Tensor, not a list
        #     orig_imgs = ops.convert_torch2numpy_batch(orig_img)

        for p in reversed(pred):
            p = self.scale_boxes(img.shape[2:], p, orig_img.shape)
            c, conf = int(p[5]), float(p[4])
            name = self.model.names[c]
            label = f'{name} {conf:.2f}'

            im_c, im_lw = (0, 255, 127), max(round(sum(orig_img.shape) / 2 * 0.003), 2)
            tf, sf = max(im_lw - 1, 1), im_lw / 3
            p1, p2 = (int(p[0]), int(p[1])), (int(p[2]), int(p[3]))
            cv.rectangle(orig_img, p1, p2, im_c, thickness=im_lw, lineType=cv.LINE_AA)

            w, h = cv.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv.rectangle(orig_img, p1, p2, im_c, -1, cv.LINE_AA)  # filled
            cv.putText(orig_img,
                       label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                       0,
                       sf,
                       (255, 255, 255),
                       thickness=tf,
                       lineType=cv.LINE_AA)

        return orig_img


if __name__ == '__main__':
    # data = {"mode": "image", "path": "source/test.jpg"}
    data = {"mode": "video", "path": "source/test.mp4"}
    net = MyNet('model/model.pt')
    net(data)
