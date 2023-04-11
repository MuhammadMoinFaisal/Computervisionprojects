import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from PIL import Image
import base64
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy
from keras.models import model_from_json
import os
import socket
import base64
# Define the Flask APP
app=Flask(__name__)



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    """
    :param filename: 
    :return: 
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
@app.route('/', methods=['POST'])
def process():
    file = request.files['image']

    img = Image.open(file.stream)
    img=img.resize((48,48))
    img = img_to_array(img)
    img_pixels = np.expand_dims(img, axis=0)
    img_pixels /= 255.0
    #model = load_model('model.h5')
    json_file = open(os.path.join(os.path.dirname(__file__), 'modelcomplete.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights and them to model
    model.load_weights(os.path.join(os.path.dirname(__file__), 'modelcomplete.h5'))
    img_class = model.predict(img_pixels)

    predictions = model.predict(img_pixels)
    max_index = int(np.argmax(predictions))

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    prediction = emotions[max_index]
    print("Class: ", prediction)
    output = np.array(prediction).tolist()
    return jsonify(output)



# This is example for Exspanse Server can check API ready or not
@app.route("/ping")
def ping():    
    return ""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)