import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import load_data, tf_dataset
from testing_training1 import iou

# Define the IoU metric function
def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou_value = intersection / union
    return iou_value

# Load the pre-trained TensorFlow model for polyp detection
def load_model(model_path):
    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model(model_path)
    return model

# Read and preprocess the input image
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x

# Read and preprocess the mask image
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

# Parse the mask image
def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

# Process the uploaded image
def process_image(file_path):
    try:
        image = read_image(file_path)
        return {'success': True, 'message': 'Image processed successfully', 'image': image.tolist()}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Process polyp detection
def process_polyp_detection(file_path):
    try:
        model = load_model(r'projects\files\model.h5')  # Update with your model path
        image = read_image(file_path)
        polyp_probability = model.predict(np.expand_dims(image, axis=0))[0]
        threshold = 0.5
        polyp_detected = (polyp_probability > threshold).any()
        
        if polyp_detected:
            return {'success': True, 'message': 'Polyp detected', 'polyp_detected': True}
        else:
            return {'success': True, 'message': 'Polyp not detected', 'polyp_detected': False}
    except Exception as e:
        return {'success': False, 'error': str(e)}
