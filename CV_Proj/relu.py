import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import initializers

training_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/shoko_o/Desktop/CV_Proj/archive/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/shoko_o/Desktop/CV_Proj/archive/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

cnn = tf.keras.models.Sequential()

##Building Convolutional Layer (ReLU)
cnn.add(tf.keras.Input(shape=(64, 64, 3)))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Dropout(0.5)) #To avoid overfitiing 
cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='glorot_uniform')) #Neurons 
cnn.add(tf.keras.layers.Dense(units=36, activation='softmax')) #Output layer

##Compiling and Training Phase 
cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=30)

##Saving Model
cnn.save('trained_model_relu.keras')


##Recording History in json
import json
with open('training_history_relu.json', 'w') as f:
    json.dump(training_history.history,f)
print(training_history.history.keys())

##Calculating Accuracy of Model Achieved on Validation Set
print("Validaion set Accuracy: {} %".format(training_history.history['val_accuracy'][-1] * 100))

