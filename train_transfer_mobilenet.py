import os
import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, save_model_dir, NUM_CLASSES, save_every_n_epoch
from prepare_data import generate_datasets, load_and_preprocess_image
import math
from models import mobilenet_v1, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, \
    efficientnet, resnext, inception_v4, inception_resnet_v1, inception_resnet_v2, \
    se_resnet, squeezenet, densenet, shufflenet_v2, resnet, se_resnext
import matplotlib.pyplot as plt
from tensorboard import notebook
import pandas as pd
import tensorflow_hub as hub


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features, data_augmentation):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()

    return images, labels


if __name__ == '__main__':
    print(tf.__name__, ": ", tf.__version__, sep="")
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # get the dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL,
                                       input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    feature_extractor.trainable = False
    model = tf.keras.Sequential([feature_extractor,
                                 tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")])

    checkpoint_save_path = "./saved_model/epoch-50"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    #
    # model_save_path = "./saved_model/epoch-50.index"
    # if os.path.exists(model_save_path):
    #     print('-------------load the model-----------------')
    #     model.load_weights(filepath=model_save_path)

    print_model_summary(network=model)

    # define loss and optimizer
    # loss_object = tf.keras.losses.sparse_categorical_crossentropy
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)
    # optimizer = tf.keras.optimizers.RMSprop()
    optimizer = tf.keras.optimizers.Adadelta()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    # @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    history = {}
    history["accuracy"] = []
    history["val_accuracy"] = []

    # start training
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for features in train_dataset:
            step += 1
            images, labels = process_features(features, data_augmentation=True)
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch,
                                                                                     EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / BATCH_SIZE),
                                                                                     train_loss.result().numpy(),
                                                                                     train_accuracy.result().numpy()))
            history["accuracy"].append(train_accuracy.result().numpy())

        for features in valid_dataset:
            valid_images, valid_labels = process_features(features, data_augmentation=False)
            valid_step(valid_images, valid_labels)
            history["val_accuracy"].append(valid_accuracy.result().numpy())

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                  EPOCHS,
                                                                  train_loss.result().numpy(),
                                                                  train_accuracy.result().numpy(),
                                                                  valid_loss.result().numpy(),
                                                                  valid_accuracy.result().numpy()))




        if epoch % save_every_n_epoch == 0:
            # Save the weights
            model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')


    # save weights
    model.save_weights(filepath=save_model_dir+"model", save_format='tf')

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()
    # save the whole model
    # tf.saved_model.save(model, save_model_dir)

    # convert to tensorflow lite format
    # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)

