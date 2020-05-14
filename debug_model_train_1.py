from conf.config import Config
from PIL import Image
import numpy as np
import os
import math
import string
import tensorflow as tf
from tensorflow import keras


class Convertion:
    @staticmethod
    def convert_to_gray(img):
        # Gray = R*0.299 + G*0.587 + B*0.114
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            # gray = np.mean(img, -1)
            return gray
        else:
            return img

    @staticmethod
    def convert_to_vector(text, captcha_len, character):
        vector = np.zeros([captcha_len, len(character)])
        for i, c in enumerate(text):
            idx = character.index(c)
            vector[i][idx] = 1.0
        return vector

    @staticmethod
    def convert_to_text(vector, character):
        text = ""
        for i, c in enumerate(vector):
            text += character[c]
        return text


def get_sample_batch(batch_size=500):
    batch_x = np.zeros([batch_size, 70, 130, 1])
    batch_y = np.zeros([batch_size, 4, 36])
    sample_dir = os.path.join(os.getcwd(), "img", "train")
    file_names = os.listdir(sample_dir)
    for i in range(500):
        captcha_image = np.array(Image.open(os.path.join(sample_dir, file_names[i])))
        captcha_text = file_names[i].split(".png")[0]
        image = tf.reshape(Convertion.convert_to_gray(captcha_image), (70, 130, 1))
        batch_x[i, :] = image
        batch_y[i, :] = Convertion.convert_to_vector(captcha_text, 4, chars)
    return batch_x, batch_y


if __name__ == "__main__":
    env_config = Config.load_env()
    captcha_str_length = env_config["captcha_length"]
    chars = ""
    if env_config["captcha_has_number"]:
        chars += string.digits
    if env_config["captcha_has_lowercase"]:
        chars += string.ascii_lowercase
    if env_config["captcha_has_uppercase"]:
        chars += string.ascii_uppercase


    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(64, (5, 5)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4 * 36))
    model.add(tf.keras.layers.Reshape([4, 36]))
    model.add(tf.keras.layers.Softmax())

    model.compile(optimizer='Adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')

    train_images, train_labels = get_sample_batch()
    model.fit(train_images, train_labels, epochs=4)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    verify_x = np.zeros([1, 70, 130, 1])
    image = np.array(Image.open(os.path.join(os.getcwd(), "img", "train", "0a3v.png")))
    image = tf.reshape(Convertion.convert_to_gray(image), (70, 130, 1))
    verify_x[0, :] = image
    predictions = probability_model.predict(verify_x)
    print(predictions)
    print(predictions[0])
    print(np.argmax(predictions[0]))


