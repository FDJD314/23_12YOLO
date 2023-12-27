from typing import Union
from pathlib import Path
import threading
import cv2 as cv
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.nn import attempt_load_one_weight
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import checks, callbacks, ops
from ultralytics.utils.torch_utils import select_device
from torchsummary import summary


class MyNet:
    def __init__(self, modelfile: Union[str, Path]):
        self.dnn = False
        self.data = 'coco.yaml'
        self.half = False
        self.device = 'cpu'
        verbose = False
        self.callbacks = callbacks.get_default_callbacks()
        self._lock = threading.Lock()
        self.conf = 0.25
        self.iou = 0.7
        self.agnostic_nms = False
        self.max_det = 300
        self.classes = None
        self.path = None
        self.imgsz = [640, 640]

        self.model = torch.load(modelfile)

        self.model.eval()
        # self.warmup()

    def __call__(self, data=None):
        self.path = data["path"]
        if data["mode"] == "image":
            img_r = self.run(cv.imread(self.path))[0].plot()
            cv.imwrite("tmp/results.jpg", img_r)
        elif data["mode"] == "video":
            # self.run()

            cap = cv.VideoCapture(data["path"])
            fourcc, fps = cap.get(cv.CAP_PROP_FOURCC), cap.get(cv.CAP_PROP_FPS)
            h, w = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            fourcc = cv.VideoWriter.fourcc(*"mp4v")
            video = cv.VideoWriter("results.mp4", fourcc, fps, (w, h))

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                img_r = self.run(frame)[0].plot()
                video.write(img_r)

            cap.release()
            video.release()

    def warmup(self, imgsz=(1, 3, 640, 640)):
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        for _ in range(2 if self.jit else 1):
            self.forward(im)  # warmup

    def run(self, im0s):
        with self._lock:
            im = self.preprocess(im0s)
            preds = self.model(im)
            results = self.postprocess(preds, im, [im0s])
        return results

    # def resize(self, img):
    #     c_shape = img.shape[:2]
    #     r = min(self.imgsz[0] / c_shape[0], self.imgsz[1] / c_shape[1])
    #
    #     new_unpad = int(round(c_shape[0] * r)), int(round(c_shape[1] * r))
    #     # dw, dh = self.imgsz[1] - new_unpad[1], self.imgsz[0] - new_unpad[0]  # wh padding
    #     # dw, dh = np.mod(dw, 32)/2, np.mod(dh, 32)/2  # wh padding
    #
    #     if c_shape[::-1] != new_unpad:  # resize
    #         img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    #     # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    #     # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    #     # img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT,
    #     #                         value=(114, 114, 114))  # add border
    #     return img
    #
    # def preprocess(self, im):
    #     not_tensor = not isinstance(im, torch.Tensor)
    #     if not_tensor:
    #         im = self.resize(im)
    #         im = im.transpose((2, 1, 0))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    #         im = np.ascontiguousarray(im)  # contiguous
    #         im = torch.from_numpy(im)
    #
    #     im = im.to(self.device)
    #     im = im.float()
    #     if not_tensor:
    #         im /= 255  # 0 - 255 to 0.0 - 1.0
    #     return torch.unsqueeze(im, dim=0)

    def pre_transform(self, im):
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(auto=True)
        return [letterbox(image=x) for x in im]

    def preprocess(self, im):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.conf,
                                        self.iou,
                                        agnostic=self.agnostic_nms,
                                        max_det=self.max_det,
                                        classes=self.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=self.path, names=self.model.names, boxes=pred))
        return results


if __name__ == '__main__':
    data = {"mode": "image", "path": "source/test.jpg"}
    # data = {"mode": "video", "path": "test.mp4"}
    # net = MyNet('yolov8n.pt')
    net = MyNet('model/model.pt')
    net(data)
