import base64
import numpy as np 
import io
import keras
import tensorflow as tf
from PIL import Image
from keras import backend as K 
from keras.models import Sequential , load_model
from keras.preprocessing.image import ImageDataGenerator , img_to_array

from flask import request , jsonify , Flask


app = Flask(__name__)


def get_model():
	global model,graph
	model = load_model("model_2.h5")
	model._make_predict_function()
	print(" * Model loaded!")
	

def preprocess_image(image ):
	if image.mode != "RGB":
		image = image.convert("RGB")
		#image = image.resize(target_size)
	image = np.array(image)
	image = np.expand_dims(image , axis=0)
		
	return image


print(" * Loading Keras model")
get_model()
graph = tf.get_default_graph()


@app.route("/predict" , methods=['POST'])
def predict():
	message = request.get_json(force=True)
	encoded= message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(image)
	with graph.as_default():
		prediction = model.predict(processed_image).tolist()

	response = {
		'prediction': {
		'EOSINOPHIL': prediction[0][3],
		'LYMPHOCYTE': prediction[0][1],
		'MONOCYTE'  : prediction[0][2],
		'NEUTROPHIL': prediction[0][0]
		}
	}

	return jsonify(response)