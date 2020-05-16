from utils.convert import NewConvert as nc
from conf.config import Config
from generate_captcha import generate_captcha
from PIL import Image
import numpy as np
import tensorflow as tf
import string
import os


def get_sample_data_from_file(path="train", size=500, section=0, height=70, width=130, cap_len=4, characters=""):
    sample_dir = os.path.join(os.getcwd(), "img", path)
    file_names = os.listdir(sample_dir)
    sample_x = np.zeros([size, height, width, 1])
    sample_y = np.zeros([size, cap_len, len(characters)])
    for seq in range(size):
        index = size*section + seq
        captcha_image = np.array(Image.open(os.path.join(sample_dir, file_names[index])))
        captcha_text = file_names[index].split(".png")[0]
        image = tf.reshape(nc.convert_to_gray(captcha_image), (70, 130, 1))
        sample_x[seq, :] = image
        sample_y[seq, :] = nc.convert_to_vector(captcha_text, cap_len, characters)
    return sample_x, sample_y


def get_sample_data_by_generator(size=500, height=70, width=130, cap_len=4, characters=""):
    sample_x = np.zeros([size, height, width, 1])
    sample_y = np.zeros([size, cap_len, len(characters)])
    for seq in range(size):
        captcha_image, captcha_text = generate_captcha(characters, cap_len, width, height)
        captcha_image = np.array(captcha_image)
        image = tf.reshape(nc.convert_to_gray(captcha_image), (70, 130, 1))
        sample_x[seq, :] = image
        sample_y[seq, :] = nc.convert_to_vector(captcha_text, cap_len, characters)
    return sample_x, sample_y


def get_model():
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

    return model


def train(model, images, labels):
    model.compile(optimizer='Adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    model.fit(images, labels, epochs=4)


if __name__ == "__main__":
    env_config = Config.load_env()
    captcha_str_length = env_config["captcha_length"]
    img_height = env_config["img_height"]
    img_width = env_config["img_width"]
    chars = ""
    if env_config["captcha_has_number"]:
        chars += string.digits
    if env_config["captcha_has_lowercase"]:
        chars += string.ascii_lowercase
    if env_config["captcha_has_uppercase"]:
        chars += string.ascii_uppercase

    model_save_path = os.path.join(os.getcwd(), "model")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_dir = "train"
    test_dir = "test"
    # sample_images, sample_labels = get_sample_data_from_file(train_dir, size=500, section=1, height=img_height, width=img_width, cap_len=captcha_str_length, characters=chars)

    new_model = get_model()
    for loop in range(1000):
        sample_images, sample_labels = get_sample_data_by_generator(size=500, height=img_height,
                                                                    width=img_width, cap_len=captcha_str_length,
                                                                    characters=chars)
        print("loop: ", loop+1, ", image: ", sample_images.shape, ", labels: ", sample_labels.shape)
        train(new_model, sample_images, sample_labels)
        if loop % 10 == 0:
            new_model.save("./model")
            print("save model when loop %d" % loop)

    print("try to predict one captcha file")
    single_image, single_text = get_sample_data_from_file("test", 1, characters=chars)
    print(np.argmax(single_text, axis=2))
    print("actual text: ", nc.convert_to_text(np.argmax(single_text, axis=2)[0], chars))
    prediction = new_model.predict(single_image)
    print(np.argmax(prediction, axis=2))
    print("predict text: ", nc.convert_to_text(np.argmax(prediction, axis=2)[0], chars))
