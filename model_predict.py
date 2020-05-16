from model_train import get_sample_data_from_file
from utils.convert import NewConvert as nc
from conf.config import Config
import numpy as np
import tensorflow as tf
import string


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

    train_dir = "train"
    test_dir = "test"
    file_amount = 5
    sample_images, sample_labels = get_sample_data_from_file(test_dir, size=file_amount, section=0, height=img_height, width=img_width, cap_len=captcha_str_length, characters=chars)
    new_model = tf.keras.models.load_model("./model")
    print(f"====Start to predict {file_amount} captcha image files====")
    label_vectors = np.argmax(sample_labels, axis=2)
    prediction = np.argmax(new_model.predict(sample_images), axis=2)
    for seq in range(file_amount):
        actual_captcha = nc.convert_to_text(label_vectors[seq], chars)
        predict_captcha = nc.convert_to_text(prediction[seq], chars)
        print(f"actual text: {actual_captcha}; predict text: {predict_captcha}")
