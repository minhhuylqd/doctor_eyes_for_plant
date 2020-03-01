from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from PIL import Image
from algorithm import create_model
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from keras.applications.resnet50 import preprocess_input
import cv2
import sys

#from utils import label_map_util
#from utils import visualization_utils as vis_util

#from Object_detection_image import process

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


sys.path.append("/home/huy/Project/ObjectDetection/models/research/")
sys.path.append("/home/huy/Project/ObjectDetection/models/research/object_detection")
sys.path.append("/home/huy/Project/ObjectDetection/models/research/object_detection/utils")

@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html")


@app.route('/uploads/<filename>')
def upload_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/result', methods=['POST'])
def result():
	#Create static - DONE
	os.system("python3 Object_detection_video.py")
	
	return render_template('resultVid.html')

	# return render_template('index.html')


if __name__ == '__main__':
	#app.run(debug=True)
	app.run(host='0.0.0.0',port="5001")
	app.run(extra_files=['templates/index.html'])
	app.config['TEMPLATES_AUTO_RELOAD'] = True

if app.config["DEBUG"]:
	@app.after_request
	def after_request(response):
		response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
		response.headers["Expires"] = 0
		response.headers["Pragma"] = "no-cache"
		return response
