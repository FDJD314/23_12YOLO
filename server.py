from flask import Flask, request, Response, send_from_directory, send_file
import numpy as np
import cv2 as cv
import os
import app_net

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def dataproc():
    if request.method == 'POST':
        cookie = request.headers['Cookie'].split(';')
        if cookie[1] == 'type=image':
            if cookie[0] == 'name=lz':
                f = request.files.get('file')
                im = cv.imdecode(np.frombuffer(f.read(), np.uint8), cv.IMREAD_COLOR)
                data = {"mode": "np", "path": im}
            else:
                f = request.data
                im = cv.imdecode(np.frombuffer(f, np.uint8), cv.IMREAD_COLOR)
                im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
                data = {"mode": "np", "path": im}
            im_r = net(data)
            _, img_encode = cv.imencode('.jpg', im_r)
            img_bytes = img_encode.tobytes()
            r = Response()
            r.headers['Content-Type'] = 'image/jpeg'
            r.data = img_bytes
            return r
        else:
            f = request.files.get('file')
            f.save("tmp/recevie.mp4")
            data = {"mode": "video", "path": "tmp/recevie.mp4"}
            # resultPath = net(data)
            # return send_file(resultPath)
            return send_file("tmp/results.mp4", mimetype='application/x-gzip')
    else:
        # return send_file(os.path.join(app.root_path, 'source', "app-release.apk"), mimetype='application/x-gzip',
        #                  as_attachment=True)
        return send_file("tmp/results.mp4", mimetype='application/x-gzip')


if __name__ == '__main__':
    net = app_net.MyNet('model/model.pt')
    app.run(host='0.0.0.0', port=80)
