import os
import glob
import tensorflow as tf
import pandas as pd
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import vgg16

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load model
new_folder = r'C:\ocular-disease-intelligent-recognition-deep-learning-master\test_run\9bcba1fa67208ad39c67a7507ea49cfc'
model = tf.keras.models.load_model(os.path.join(new_folder, 'model_weights.h5'))

IMAGE_SIZE = 224

# Predict & classify image
def classify(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = vgg16.preprocess_input(img_array)  # Preprocess input
    prob = model.predict(img_array)
    class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
    label = class_names[tf.argmax(prob[0])]
    classified_prob = tf.reduce_max(prob) * 100
    # Collect all class name predictions along with their probabilities
    predictions = {class_names[i]: round(prob[0][i].item() * 100, 2) for i in range(len(class_names))}
    return label, classified_prob, predictions

# Route for home page
@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")

# Route for uploading and classifying images
@app.route("/classify", methods=["POST"])
def upload_and_classify():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)

    label, prob, predictions = classify(upload_image_path)
    prob = round(float(prob), 2)

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob, predictions=predictions
    )

# Route for serving uploaded images
@app.route("/uploads/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
