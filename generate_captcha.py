from captcha.image import ImageCaptcha
import random
import string
import os


def generate_save_captcha(length, amount, folder_name, width=130, height=70):
    characters = string.digits + string.ascii_lowercase
    generator = ImageCaptcha(width=width, height=height)
    img_path = os.path.join(os.getcwd(), "img", folder_name)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for _ in range(amount):
        captcha_str = "".join([random.choice(characters) for j in range(length)])
        img = generator.create_captcha_image(captcha_str, (65, 105, 225), (255, 255, 255))
        img.save(os.path.join(img_path, "%s.png" % captcha_str))


if __name__ == "__main__":
    train_img_amount = 60000
    test_img_amount = 10000
    captcha_str_length = 4
    generate_save_captcha(captcha_str_length, train_img_amount, "train")
    generate_save_captcha(captcha_str_length, test_img_amount, "test")
