from conf.config import Config
from PIL import Image
import numpy as np
import os
import math
import string
import tensorflow as tf
from tensorflow import keras


def convert_to_gray(img):
    # Gray = R*0.299 + G*0.587 + B*0.114
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        # gray = np.mean(img, -1)
        return gray
    else:
        return img


def convert_to_vector(text, captcha_len, character):
    vector = np.zeros(captcha_len * len(character))
    for i, c in enumerate(text):
        index = i * len(character) + character.index(c)
        vector[index] = 1
    return vector


def get_sample(folder_name, split=10, start=10):
    sample_dir = os.path.join(os.getcwd(), "img", folder_name)
    file_names = os.listdir(sample_dir)
    file_amount = len(file_names)
    gap = [0]
    amount_split = math.ceil(file_amount/split)
    for i in range(split):
        current_amount = amount_split*(i+1)
        if current_amount > file_amount:
            gap.append(file_amount)
        else:
            gap.append(amount_split*(i+1))
    print(gap)
    print(gap[start-1: start+1])
    batch_images = []
    for seq in range(gap[start-1], gap[start]):
        batch_images[seq, :] = np.array(Image.open(os.path.join(sample_dir, file_names[seq])))
        print(batch_images)
        input("-=-=-=")
    return file_amount


def get_sample_batch(batch_size = 500):
    batch_x = np.zeros([batch_size, 130*70])
    batch_y = np.zeros([batch_size, 4*36])
    sample_dir = os.path.join(os.getcwd(), "img", "train")
    file_names = os.listdir(sample_dir)
    for i in range(500):
        batch_x[i, :] = convert_to_gray(np.array(Image.open(os.path.join(sample_dir, file_names[i])))).flatten() / 255
        # batch_x[i, :] = tf.reshape(convert_to_gray(np.array(Image.open(os.path.join(sample_dir, file_names[i])))), (130, 70, 1))
        batch_y[i, :] = convert_to_vector(file_names[i].split(".png")[0], 4, chars)
    return batch_x, batch_y


if __name__ == "__main__":
    # captcha_image = Image.open(os.path.join(os.getcwd(), "img", "train", "0a2p.png"))
    # image = np.array(captcha_image)
    # convert = convert_to_gray(image)

    env_config = Config.load_env()
    captcha_str_length = env_config["captcha_length"]
    chars = ""
    if env_config["captcha_has_number"]:
        chars += string.digits
    if env_config["captcha_has_lowercase"]:
        chars += string.ascii_lowercase
    if env_config["captcha_has_uppercase"]:
        chars += string.ascii_uppercase

    # print(convert_to_vector("9abc", captcha_str_length, chars))

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

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    train_images, train_labels = get_sample_batch()
    model.fit(train_images, train_labels, epochs=10)
    print("y预测=\n", np.argmax(model.predict(batch_x), axis=2))
    print("y实际=\n", np.argmax(batch_y, axis=2))

