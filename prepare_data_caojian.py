import tensorflow as tf
import pathlib
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    BATCH_SIZE, train_tfrecord, valid_tfrecord, test_tfrecord
from parse_tfrecord import get_parsed_dataset


def load_and_preprocess_image(image_raw, data_augmentation=False):
    # decode
    image_tensor = tf.io.decode_image(contents=image_raw, channels=CHANNELS, dtype=tf.dtypes.float32)

    if data_augmentation:
        image = tf.image.random_flip_left_right(image=image_tensor)
        image = tf.image.resize_with_crop_or_pad(image=image,
                                                 target_height=int(IMAGE_HEIGHT * 1.4),
                                                 target_width=int(IMAGE_WIDTH * 1.4))
        # image = tf.image.random_crop(value=image, size=[IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
        # image = tf.image.random_brightness(image=image, max_delta=0.1)
    else:
        image = tf.image.resize(image_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = image / 255.0

    return image


def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_the_length_of_dataset(dataset):
    count = 0
    for i in dataset:
        count += 1
    return count


def generate_datasets():
    train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
    valid_dataset = get_parsed_dataset(tfrecord_name=valid_tfrecord)
    test_dataset = get_parsed_dataset(tfrecord_name=test_tfrecord)

    train_count = get_the_length_of_dataset(train_dataset)
    valid_count = get_the_length_of_dataset(valid_dataset)
    test_count = get_the_length_of_dataset(test_dataset)


    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count
