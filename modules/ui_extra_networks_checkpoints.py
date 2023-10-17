import html
import json
import os
from modules import shared, ui_extra_networks, sd_models


class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Model')

    def refresh(self):
        shared.refresh_checkpoints()

    def list_items(self):
        checkpoint: sd_models.CheckpointInfo
        checkpoints = sd_models.checkpoints_list.copy()
        for name, checkpoint in checkpoints.items():
            try:
                fn = os.path.splitext(checkpoint.filename)[0]
                record = {
                    "type": 'Model',
                    "name": checkpoint.name,
                    "title": checkpoint.title,
                    "filename": checkpoint.filename,
                    "hash": checkpoint.shorthash,
                    "search_term": self.search_terms_from_path(checkpoint.title),
                    "preview": self.find_preview(fn),
                    "local_preview": f"{fn}.{shared.opts.samples_format}",
                    "description": self.find_description(fn),
                    "info": self.find_info(fn),
                    "metadata": checkpoint.metadata,
                    "onclick": '"' + html.escape(f"""return selectCheckpoint({json.dumps(name)})""") + '"',
                }
                yield record
            except Exception as e:
                shared.log.debug(f"Extra networks error: type=model file={name} {e}")

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.ckpt_dir, shared.opts.diffusers_dir, sd_models.model_path] if v is not None]
