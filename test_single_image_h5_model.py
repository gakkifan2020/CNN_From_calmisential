import tensorflow as tf
import os
import tensorflow_hub as hub
from configuration import save_model_dir, test_image_dir
from train import get_model
from prepare_data import load_and_preprocess_image


def get_class_id(image_root):
    id_cls = {}
    for i, item in enumerate(os.listdir(image_root)):
        if os.path.isdir(os.path.join(image_root, item)):
            id_cls[i] = item
    return id_cls


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = tf.keras.models.load_model('model.h5',
                                       custom_objects={'KerasLayer': hub.KerasLayer, 'Dense': tf.keras.layers.Dense},
                                       compile=False)
    # model = tf.keras.models.load_model(h5_save_path, compile=False)
    model.summary()

    image_raw = tf.io.read_file(filename=test_image_dir)
    image_tensor = load_and_preprocess_image(image_raw)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    pred = model(image_tensor, training=False)
    print(pred)
    idx = tf.math.argmax(pred, axis=-1).numpy()[0]
    print(idx)

    # id_cls = get_class_id("./original_dataset")
    #
    # print("The predicted category of this picture is: {}".format(id_cls[idx]))