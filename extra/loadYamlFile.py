import os
import yaml

from modules.paths_internal import sd_configs_path


class ExtraConfig:
    def __init__(self, environment=None):
        if environment is not None:
            environment = f"-{environment}"
        else:
            environment = ""
        self.file_path = os.path.join(sd_configs_path, f"sd-extra{environment}.yaml")
        self.config = None

    def load_config(self):
        with open(self.file_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_config(self):
        if not self.config:
            self.load_config()
        return self.config
