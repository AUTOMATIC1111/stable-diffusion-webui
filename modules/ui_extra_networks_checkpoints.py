import html
import json
import os
import urllib.parse

from modules import shared, ui_extra_networks, sd_models


class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Checkpoints')

    def refresh(self):
        shared.refresh_checkpoints()

    def list_items(self):
        for name, checkpoint1 in sd_models.checkpoints_list.items():
            checkpoint: sd_models.CheckpointInfo = checkpoint1
            path, ext = os.path.splitext(checkpoint.filename)
            previews = [path + ".png", path + ".preview.png"]

            preview = None
            for file in previews:
                if os.path.isfile(file):
                    preview = self.link_preview(file)
                    break

            yield {
                "name": checkpoint.model_name,
                "filename": path,
                "preview": preview,
                "onclick": '"' + html.escape(f"""return selectCheckpoint({json.dumps(name)})""") + '"',
                "local_preview": path + ".png",
            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.ckpt_dir, sd_models.model_path]

