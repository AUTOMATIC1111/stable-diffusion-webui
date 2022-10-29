import json
import os

from modules import paths


class TrainConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = None
        self.model_name = None
        self.scheduler = None
        self.src = None
        self.total_steps = None
        self.__dict__ = self

    def create_new(self, name, scheduler, src, total_steps):
        self.name = name
        self.model_name = name
        self.scheduler = scheduler
        self.src = src
        self.total_steps = total_steps
        return self

    def from_ui(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        return self.__dict__

    def from_file(self, model_name):
        model_path = paths.models_path
        model_dir = os.path.join(model_path, "dreambooth", model_name, "working")
        config_file = os.path.join(model_dir, "config.json")
        try:
            with open(config_file, 'r') as openfile:
                config = json.load(openfile)
                for key in config:
                    value = config[key]
                    self.__dict__[key] = value
        except Exception as e:
            print(f"Exception loading config: {e}")
            return None
            pass
        return self.__dict__

    def save(self):
        model_path = paths.models_path
        model_dir = os.path.join(model_path, "dreambooth", self.__dict__["model_name"], "working")
        config_file = os.path.join(model_dir, "config.json")
        config = json.dumps(self.__dict__)
        with open(config_file, "w") as outfile:
            outfile.write(config)
