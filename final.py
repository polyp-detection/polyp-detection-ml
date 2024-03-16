import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import load_data, tf_dataset
from testing_training1 import iou
import pickle

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if _name_ == "_main_":
    ## Dataset
    path = "Polyps_dataset"
    valid_size = 0.2
    test_size = 0.2
    batch_size = 8
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path, split=valid_size)

    # Calculate the actual number of samples for validation and testing
    valid_samples = int(len(train_x) * valid_size)
    test_samples = int(len(train_x) * test_size)

    # Split the training set into training, validation, and testing sets
    train_x, valid_x, test_x = (
        train_x[:-valid_samples - test_samples],
        train_x[-valid_samples - test_samples:-test_samples],
        train_x[-test_samples:]
    )
    train_y, valid_y, test_y = (
        train_y[:-valid_samples - test_samples],
        train_y[-valid_samples - test_samples:-test_samples],
        train_y[-test_samples:]
    )

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model(r"files\model.h5")

    model.evaluate(test_dataset, steps=test_steps)

    # Pickle the model
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results/{i}.png", image)