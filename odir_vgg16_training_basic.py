# Copyright 2019-2020 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tensorflow as tf
from odir_model_factory import Factory, ModelTypes
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import secrets
import odir
from odir_advance_plotting import Plotter
from odir_kappa_score import FinalScore
from odir_predictions_writer import Prediction
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16

batch_size = 64
num_classes = 8
epochs = 50
patience = 5

token = secrets.token_hex(16)
folder = r'C:\ocular-disease-intelligent-recognition-deep-learning-master\test_run'

new_folder = os.path.join(folder, token)

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

factory = Factory((224, 224, 3), defined_metrics)
model = factory.compile(ModelTypes.vgg16)

(x_train, y_train), (x_test, y_test) = odir.load_data(224)

x_test_drawing = x_test

x_train = vgg16.preprocess_input(x_train)
x_test = vgg16.preprocess_input(x_test)

from tensorflow.keras.utils import to_categorical

# Assuming y_train and y_test are your target arrays
y_train_encoded = to_categorical(y_train, num_classes=8)
y_test_encoded = to_categorical(y_test, num_classes=8)

class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

# plot data input
plotter = Plotter(class_names)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)

class_weight = {0: 1.,
                1: 1.583802025,
                2: 8.996805112,
                3: 10.24,
                4: 10.05714286,
                5: 1.,
                6: 1.,
                7: 2.505338078}

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True, #class_weight = class_weight,
                    validation_data=(x_test, y_test), callbacks=[callback])

print("saving")
model.save(os.path.join(new_folder, 'model_weights.h5'))

print("plotting")
plotter.plot_metrics(history, os.path.join(new_folder, 'plot1.png'), 2)

# Hide meanwhile for now
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig(os.path.join(new_folder, 'plot2.png'))
plt.show()

# display the content of the model
baseline_results = model.evaluate(x_test, y_test, verbose=2)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

# test a prediction
test_predictions_baseline = model.predict(x_test)
plotter.plot_confusion_matrix_generic(y_test, test_predictions_baseline, os.path.join(new_folder, 'plot3.png'), 0)

# save the predictions
prediction_writer = Prediction(test_predictions_baseline, 400, new_folder)
prediction_writer.save()
prediction_writer.save_all(y_test)

# show the final score
score = FinalScore(new_folder)
score.output()

# plot output results
plotter.plot_output(test_predictions_baseline, y_test, x_test_drawing, os.path.join(new_folder, 'plot4.png'))
