import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from tensorflow import keras
import numpy as np


def createModel():
    xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)
    model = keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[1])
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    model.fit(xs, ys, epochs=400)

    model.save('model')



def pb2h5():
    pb_model_dir = "./model"
    h5_model = "./mymodel.h5"

    # Loading the Tensorflow Saved Model (PB)
    model = tf.keras.models.load_model(pb_model_dir)
    print(model.summary())

    # Saving the Model in H5 Format
    tf.keras.models.save_model(model, h5_model)

    # Loading the H5 Saved Model
    loaded_model_from_h5 = tf.keras.models.load_model(h5_model)
    print(loaded_model_from_h5.summary())

createModel()
pb2h5()
