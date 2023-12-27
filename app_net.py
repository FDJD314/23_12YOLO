from typing import Union
from pathlib import Path
import threading
import cv2 as cv
import numpy as np
import torch
from ultralytics.utils import ops


class Colors:
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class MyNet:
    def __init__(self, modelfile: Union[str, Path]):
        self.dnn = False
        self.data = 'coco.yaml'
        self.half = False
        self.device = 'cpu'
        verbose = False
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
            img_r = self.run(cv.imread(self.path))
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
        # dw, dh = self.imgsz[1] - new_unpad[1], self.imgsz[0] - new_unpad[0]  # wh padding
        # dw, dh = np.mod(dw, 32)/2, np.mod(dh, 32)/2  # wh padding

        if c_shape[::-1] != new_unpad:  # resize
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT,
        #                         value=(114, 114, 114))  # add border
        return img

    def scale_boxes(self, img1_shape, boxes, img0_shape):
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
        pred = ops.non_max_suppression(preds,
                                       self.conf,
                                       self.iou,
                                       agnostic=self.agnostic_nms,
                                       max_det=self.max_det,
                                       classes=self.classes)[0]

        # if not isinstance(orig_img, list):  # input images are a torch.Tensor, not a list
        #     orig_imgs = ops.convert_torch2numpy_batch(orig_img)

        # results = []
        # for i, pred in enumerate(preds):
        #     orig_img = orig_imgs[i]
        #     pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        #     results.append(Results(orig_img, path=self.path, names=self.model.names, boxes=pred))
        # return results

        for p in reversed(pred):
            p = self.scale_boxes(img.shape[2:], p, orig_img.shape)
            c, conf = int(p[5]), float(p[4])
            name = self.model.names[c]
            label = f'{name} {conf:.2f}'

            im_c, im_lw = colors(p[5], True), max(round(sum(orig_img.shape) / 2 * 0.003), 2)
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
    data = {"mode": "image", "path": "source/test.jpg"}
    # data = {"mode": "video", "path": "test.mp4"}
    # net = MyNet('yolov8n.pt')
    net = MyNet('model/model.pt')
    net(data)
