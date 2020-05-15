from captcha.image import ImageCaptcha
from conf.config import Config
import random
import string
import os


def generate_save_captcha(characters, length, amount, folder_name, width=130, height=70):
    generator = ImageCaptcha(width=width, height=height)
    img_path = os.path.join(os.getcwd(), "img", folder_name)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for _ in range(amount):
        captcha_str = "".join([random.choice(characters) for j in range(length)])
        img = generator.create_captcha_image(captcha_str, (65, 105, 225), (255, 255, 255))
        img.save(os.path.join(img_path, "%s.png" % captcha_str))


def generate_captcha(characters, length, width=130, height=70):
    generator = ImageCaptcha(width=width, height=height)
    captcha_str = "".join([random.choice(characters) for j in range(length)])
    img = generator.create_captcha_image(captcha_str, (65, 105, 225), (255, 255, 255))
    return img, captcha_str


if __name__ == "__main__":
    env_config = Config.load_env()
    train_img_amount = env_config["train_amount"]
    test_img_amount = env_config["test_amount"]
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
    generate_save_captcha(chars, captcha_str_length, train_img_amount, "train", img_width, img_height)
    generate_save_captcha(chars, captcha_str_length, test_img_amount, "test", img_width, img_height)
