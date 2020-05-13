from utils.yaml_helper import YamlHelper
import os


class Config:
    @classmethod
    def load_env(cls, name=None):
        if name:
            return YamlHelper.load_yaml(os.path.join(os.getcwd(), "conf", "env.yaml"))[name]
        else:
            return YamlHelper.load_yaml(os.path.join(os.getcwd(), "conf", "env.yaml"))
