import numpy as np
from PIL import Image
import os


def convert_to_gray(img):
    # Gray = R*0.299 + G*0.587 + B*0.114
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray
    else:
        return img


if __name__ == "__main__":
    captcha_image = Image.open(os.path.join(os.getcwd(), "img", "train", "0a2p.png"))
    img = np.array(captcha_image)
