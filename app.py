import os
import glob
import tensorflow as tf
import pandas as pd
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import inception_v3

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load model
new_folder = r'C:\ocular-disease-intelligent-recognition-deep-learning-master\test_run\d3bbdd18f8b56e2509419b9bd7291fee'
model = tf.keras.models.load_model(os.path.join(new_folder, 'model_weights.h5'))

IMAGE_SIZE = 224

# Predict & classify image
def classify(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = inception_v3.preprocess_input(img_array)  # Preprocess input
    prob = model.predict(img_array)
    class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
    label = class_names[tf.argmax(prob[0])]
    classified_prob = tf.reduce_max(prob) * 100
    # Collect all class name predictions along with their probabilities
    predictions = {class_names[i]: round(prob[0][i].item() * 100, 2) for i in range(len(class_names))}
    return label, classified_prob, predictions


# home page
@app.route("/", methods=['GET'])
def home():
    filelist = glob.glob("uploads/*.*")
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")
    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_image_path)

        label, prob, predictions = classify(upload_image_path)
        prob = round(float(prob), 2)

        # Get the filename from the uploaded file
        filename = file.filename

        # Read the ground truth data from odir_ground_truth.csv
        ground_truth = pd.read_csv(os.path.join(STATIC_FOLDER, 'odir.csv'))

        # Get the ground truth label for the uploaded filename
        ground_truth_label = ground_truth.loc[ground_truth['Filename'] == filename, 'Total'].values[0]

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob, predictions=predictions, ground_truth_label=ground_truth_label
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
