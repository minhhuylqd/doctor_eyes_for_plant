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

"""
def _process(img):
	MODEL_NAME = 'inference_graph'
	IMAGE_NAME = img
	# Grab path to current working directory
	CWD_PATH = os.getcwd()

	# Path to frozen detection graph .pb file, which contains the model that is used
	# for object detection.
	PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

	# Path to label map file
	PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

	# Path to image
	PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

	# Number of classes the object detector can identify
	NUM_CLASSES = 6

	# Load the label map.
	# Label maps map indices to category names, so that when our convolution
	# network predicts `5`, we know that this corresponds to `king`.
	# Here we use internal utility functions, but anything that returns a
	# dictionary mapping integers to appropriate string labels would be fine
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	# Load the Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

		sess = tf.Session(graph=detection_graph)

	# Define input and output tensors (i.e. data) for the object detection classifier

	# Input tensor is the image
	image_tensor = detection_graphapp.get_tensor_by_name('image_tensor:0')

	# Output tensors are the detection boxes, scores, and classes
	# Each box represents a part of the image where a particular object was detected
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

	# Each score represents level of confidence for each of the objects.
	# The score is shown on the result image, together with the class label.
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

	# Number of objects detected
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	# Load image using OpenCV and
	# expand image dimensions to have shape: [1, None, None, 3]
	# i.e. a single-column array, where each item in the column has the pixel RGB value
	image = cv2.imread(PATH_TO_IMAGE)
	image_expanded = np.expand_dims(image, axis=0)

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: image_expanded})

	# Draw the results of the detection (aka 'visulaize the results')

	vis_util.visualize_boxes_and_labels_on_image_array(
		image,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=0.60)
	return image
"""

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
	if not os.path.exists(app.config['UPLOAD_FOLDER']):
		os.mkdir(app.config['UPLOAD_FOLDER'])
	
	#Request upload file - DONE
	file = request.files['image']
	if (file.filename == ""):
		return redirect("/")

	

	filename = file.filename
	filename = secure_filename(filename)
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
	name, ext = os.path.splitext(filename)
	
	
	src = 'static/' + filename
	dst = 'static/testIMG.jpg'
	os.rename(src, dst)
	#image = pre_process(dst)
	
	img = Image.open("static/testIMG.jpg")

	os.system("python3 Object_detection_image.py")
	imageResDst = 'static/resIMG.jpg'

	with open ("static/output.txt", "r") as myfile:
		label=myfile.read()
	print(label)
	if label == "apple_healthy":
		plant_name = "Táo / Apple"
		disease = "Cây Khỏe / Healthy"
		treatment_technique1 = ""
		treatment_technique2 = ""
		treatment_technique3 = ""
		treatment_technique4 = ""

	elif label =="apple_black_rot":
		plant_name = "Táo / Apple"
		disease = "Bệnh thối đen / Black rot"
		treatment_technique1 = "Weed regularly to improve air circulation and light access around all plants affected by black rot. Remove all dead plant material from the area, placing it into plastic trash bags to prevent further spread of the disease."
		treatment_technique2 = "Nhổ cỏ thường xuyên xung quanh các cây bệnh để tăng sự tiếp xúc với không khí và ánh sáng. Nhổ bỏ tất cả các cây chết khỏi vườn, vứt chúng vào bao rác để ngăn ngừa sự lây lan."
		treatment_technique3 = "Although using fungicides is the most effective chemical way, they should be used only when the risk of black rot infection is high, what result in lower numbers of applications during the season and protection of the environment."
		treatment_technique4 = "Dù sử dụng thuốc diệt nấm là phương pháp tốt nhất, nhưng chỉ nên sử dụng chúng khi cây nhiễm bệnh thối đen quá nặng, bởi vì thuốc diệt nấm sẽ cho ra năng suất nông sản thấp trong mùa vụ và là để bảo vệ môi trường."

	elif label == "grape_black_rot":
		plant_name = "Nho / Grape"
		disease = "Bệnh thối đen / Black rot"
		treatment_technique1 = "Weed regularly to improve air circulation and light access around all plants affected by black rot. Remove all dead plant material from the area, placing it into plastic trash bags to prevent further spread of the disease."
		treatment_technique2 = "Nhổ cỏ thường xuyên xung quanh các cây bệnh để tăng sự tiếp xúc với không khí và ánh sáng. Nhổ bỏ tất cả các cây chết khỏi vườn, vứt chúng vào bao rác để ngăn ngừa sự lây lan."
		treatment_technique3 = "Although using fungicides is the most effective chemical way, they should be used only when the risk of black rot infection is high, what result in lower numbers of applications during the season and protection of the environment."
		treatment_technique4 = "Dù sử dụng thuốc diệt nấm là phương pháp tốt nhất, nhưng chỉ nên sử dụng chúng khi cây nhiễm bệnh thối đen quá nặng, bởi vì thuốc diệt nấm sẽ cho ra năng suất nông sản thấp trong mùa vụ và là để bảo vệ môi trường."

	elif label == "grape_healthy":
		plant_name = "Nho / Grape"
		disease = "Cây Khỏe / Healthy"
		treatment_technique1 = ""
		treatment_technique2 = ""
		treatment_technique3 = ""
		treatment_technique4 = ""

	elif label == "pepperbell_bacterial":
		plant_name = "Ớt chuông / Pepper Bell"
		disease = "Bệnh đốm lá vi khuẩn / Pepper Bell Bacterial"
		treatment_technique2 = "Hạn chế tối đa sử dụng hệ thống tưới phun mưa để tưới cho vườn ớt."
		treatment_technique4 = "Đảo vụ thôi trồng ớt ít nhất 1 năm."
		treatment_technique1 = ""
		treatment_technique3 = ""
		
		

	elif label == "pepperbell_healthy":
		plant_name = "Ớt chuông / Pepper Bell"
		disease = "Cây Khỏe / Healthy"
		treatment_technique1 = ""
		treatment_technique2 = ""
		treatment_technique3 = ""
		treatment_technique4 = ""

	return render_template('result.html', imageResDst = imageResDst, 
	plant_name = plant_name, disease = disease, 
	treatment_technique1 = treatment_technique1, treatment_technique2 = treatment_technique2,
	treatment_technique3 = treatment_technique3, treatment_technique4 = treatment_technique4)

	# return render_template('index.html')


if __name__ == '__main__':
	#app.run(debug=True)
	#app.run(host='0.0.0.0')
	app.run(extra_files=['templates/index.html'])
	app.config['TEMPLATES_AUTO_RELOAD'] = True

if app.config["DEBUG"]:
	@app.after_request
	def after_request(response):
		response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
		response.headers["Expires"] = 0
		response.headers["Pragma"] = "no-cache"
		return response
