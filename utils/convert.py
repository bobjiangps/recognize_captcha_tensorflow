import numpy as np


class Convert:
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
        vector = np.zeros(captcha_len*len(character))
        for i, c in enumerate(text):
            idx = character.index(c)
            vector[idx+len(character)*i] = 1
        return vector
        # vector = np.zeros([captcha_len, len(character)])
        # for i, c in enumerate(text):
        #     idx = character.index(c)
        #     vector[i][idx] = 1
        # return vector

    @staticmethod
    def convert_to_text(vector, captcha_len, character):
        text = ""
        for seq in range(captcha_len):
            for i, c in enumerate(vector[seq * len(character):(seq + 1) * len(character)]):
                if c > 0:
                    print(seq,i)
                    text += character[i]
        return text
        # text = ""
        # for i, c in enumerate(vector):
        #     text += character[c]
        # return text


class NewConvert:
    @staticmethod
    def convert_to_gray(img):
        # Gray = R*0.299 + G*0.587 + B*0.114
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
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
        # text = ""
        # for v in vector:
        #     for i, c in enumerate(v):
        #         if c > 0:
        #             text += character[i]
        #             break
        # return text
