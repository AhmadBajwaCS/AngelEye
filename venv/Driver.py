import tensorflow as tf
import cv2
import numpy as np
tf.compat.v1.disable_eager_execution()

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)

load_model()