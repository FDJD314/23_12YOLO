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


class MyNet:
    def __init__(self, modelfile: Union[str, Path]):
        # self.args = get_cfg(cfg, overrides)
        # self.done_warmup = False
        self.dnn = False
        self.data = 'coco.yaml'
        self.half = False
        self.device = 'cpu'
        verbose = True
        self.callbacks = callbacks.get_default_callbacks()
        self._lock = threading.Lock()
        self.conf = 0.25
        self.iou = 0.7
        self.agnostic_nms = False
        self.max_det = 300
        self.classes = None
        self.path = None
        self.imgsz = [640, 640]
        callbacks.add_integration_callbacks(self)

        modelfile = str(modelfile).strip()
        modelfile = checks.check_model_file_from_stem(modelfile)
        model, _ = attempt_load_one_weight(modelfile)
        self.model = AutoBackend(model,
                                 device=select_device(self.device, verbose=verbose),
                                 dnn=self.dnn,
                                 data=self.data,
                                 fp16=self.half,
                                 fuse=True,
                                 verbose=verbose)

        self.model.eval()
        self.model.warmup()

    def __call__(self, data=None):
        self.path = data["path"]
        if data["mode"] == "image":
            img_r = self.run(cv.imread(self.path))[0].plot()
            cv.imwrite("results.jpg", img_r)
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

    def run(self, im0s, *args, **kwargs):
        profilers = (ops.Profile(), ops.Profile(), ops.Profile())
        # self.run_callbacks('on_predict_start')
        with self._lock:
            # self.run_callbacks('on_predict_batch_start')

            # Preprocess
            with profilers[0]:
                im = self.preprocess([im0s])

            # Inference
            with profilers[1]:
                preds = self.model(im, augment=False, visualize=False, embed=None, *args, **kwargs)

            # Postprocess
            with profilers[2]:
                results = self.postprocess(preds, im, [im0s])

        #     self.run_callbacks('on_predict_postprocess_end')
        #     self.run_callbacks('on_predict_batch_end')
        # self.run_callbacks('on_predict_end')

        return results

    def pre_transform(self, im):
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
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

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


if __name__ == '__main__':
    # data = {"mode": "image", "path": "test.jpg"}
    data = {"mode": "video", "path": "test.mp4"}
    net = MyNet('yolov8n.pt')
    net(data)
