from flask import Flask, request, Response
import numpy as np
import cv2 as cv
import app_net

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def dataproc():
    f = request.data
    im = cv.imdecode(np.frombuffer(f, np.uint8), cv.IMREAD_COLOR)
    # cv.imwrite("webtest.jpg", im)

    im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
    data = {"mode": "np", "path": im}
    im_r = net(data)

    # im_r = cv.imread("source/test.jpg")
    _, img_encode = cv.imencode('.jpg', im_r)
    img_bytes = img_encode.tobytes()
    r = Response()
    r.headers['Content-Type'] = 'image/jpeg'
    r.data = img_bytes
    return r


if __name__ == '__main__':
    net = app_net.MyNet('model/model.pt')
    app.run(host='0.0.0.0', port=80)
