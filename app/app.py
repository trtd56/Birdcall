from flask import Flask, render_template, request, jsonify, make_response
import os
import socket
from datetime import datetime
import werkzeug
import torch

from utils import BirdcallNet, predict

app = Flask(__name__)


UPLOAD_DIR = "uploads"
model = BirdcallNet()
model.load_state_dict(torch.load("birdcallnet_f0.bin", map_location=torch.device('cpu')))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/data/upload', methods=['POST'])
def upload_multipart():

    if 'uploadFile' not in request.files:
        make_response(jsonify({'result':'uploadFile is required.'}))

    file = request.files['bird']
    fileName = file.filename
    if '' == fileName:
        make_response(jsonify({'result':'filename must not empty.'}))

    saveFileName = datetime.now().strftime("%Y%m%d_%H%M%S_") \
        + werkzeug.utils.secure_filename(fileName)
    file.save(os.path.join(UPLOAD_DIR, saveFileName))
    output = predict(os.path.join(UPLOAD_DIR, saveFileName), model)
    return make_response(output)

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8080)
