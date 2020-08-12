import tensorflow as tf
from configuration import save_pb_model_dir, test_image_dir
from prepare_data import load_and_preprocess_image
import os
import numpy as np
from cv2 import dnn
import cv2


def get_single_picture_prediction(model, picture_dir):
    image_tensor = load_and_preprocess_image(tf.io.read_file(filename=picture_dir), data_augmentation=False)
    image = tf.expand_dims(image_tensor, axis=0)
    prediction = model(image, training=False)
    pred_class = tf.math.argmax(prediction, axis=-1)
    return pred_class


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    img_file = test_image_dir + '160025.jpg'
    img_cv2 = cv2.imread(img_file)
    inWidth = 224
    inHeight = 224
    blob = cv2.dnn.blobFromImage(img_cv2,
                                 scalefactor=1.0 / 255,
                                 size=(inWidth, inHeight),
                                 mean=(0, 0, 0),
                                 swapRB=False,
                                 crop=False)
    # load the model
    export_dir = save_pb_model_dir + "model.pb"
    # tf.saved_model.load(export_dir,
    #                     tags=None,
    #                     options=None
    # )
    if os.path.exists(export_dir):
        model = dnn.readNetFromTensorflow(export_dir)
        # Run a model
        model.setInput(blob)
        out = model.forward()

        # Get a class with a highest score.
        out = out.flatten()
        classId = np.argmax(out)
        confidence = out[classId]
    print(confidence)

    # pred_class = get_single_picture_prediction(model, test_example)
    # print(pred_class)