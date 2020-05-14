import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import os
import string
import numpy as np

from utils.convert import Convert
from conf.config import Config


def get_sample_data(path, size=500, height=70, width=130, cap_len=4, characters=""):
    sample_dir = os.path.join(os.getcwd(), "img", path)
    file_names = os.listdir(sample_dir)
    sample_x = np.zeros([size, height*width])
    sample_y = np.zeros([size, cap_len*len(characters)])
    for seq in range(size):
        captcha_image = np.array(Image.open(os.path.join(sample_dir, file_names[seq])))
        captcha_text = file_names[seq].split(".png")[0]
        image = Convert.convert_to_gray(captcha_image)
        sample_x[seq, :] = image.flatten() / 255
        sample_y[seq, :] = Convert.convert_to_vector(captcha_text, cap_len, characters)
    return sample_x, sample_y


if __name__ == "__main__":
    batch_size = 128
    epochs = 15
    IMG_HEIGHT = 70
    IMG_WIDTH = 130

    env_config = Config.load_env()
    captcha_str_length = env_config["captcha_length"]
    chars = ""
    if env_config["captcha_has_number"]:
        chars += string.digits
    if env_config["captcha_has_lowercase"]:
        chars += string.ascii_lowercase
    if env_config["captcha_has_uppercase"]:
        chars += string.ascii_uppercase

    train_dir = os.path.join(os.getcwd(), "img", "train")
    test_dir = os.path.join(os.getcwd(), "img", "test")

    sample_images, sample_labels = get_sample_data(train_dir, size=500, height=IMG_HEIGHT, width=IMG_WIDTH, cap_len=captcha_str_length, characters=chars)

    input_layer = tf.keras.Input()
    x = layers.Conv2D(32, 3, activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Flatten()(x)
    # x = layers.Dense(1024, activation='relu')(x)
    # # x = layers.Dropout(0.5)(x)
    #
    # x = layers.Dense(D * N_LABELS, activation='softmax')(x)
    # x = layers.Reshape((D, N_LABELS))(x)

    model = models.Model(inputs=input_layer, outputs=x)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
