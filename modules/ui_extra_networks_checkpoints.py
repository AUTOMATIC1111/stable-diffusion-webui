import html
import json
import os
from modules import shared, ui_extra_networks, sd_models, paths

reference_dir = os.path.join(paths.models_path, 'Reference')

class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Model')

    def refresh(self):
        shared.refresh_checkpoints()

    def list_reference(self):
        if shared.backend != shared.Backend.DIFFUSERS:
            return []
        reference_models = shared.readfile(os.path.join('html', 'reference.json'))
        for k, v in reference_models.items():
            name = os.path.join(reference_dir, k)
            yield {
                "type": 'Model',
                "name": name,
                "title": name,
                "filename": v['path'],
                "search_term": self.search_terms_from_path(name),
                "preview": self.find_preview(os.path.join(reference_dir, os.path.basename(v['path']))),
                "local_preview": f"{os.path.splitext(name)[0]}.{shared.opts.samples_format}",
                "onclick": '"' + html.escape(f"""return selectReference({json.dumps(v['path'])})""") + '"',
                "hash": None,
                "mtime": 0,
                "size": 0,
                "info": {},
                "metadata": {},
                "description": v.get('desc', ''),
            }

    def list_items(self):
        checkpoint: sd_models.CheckpointInfo
        checkpoints = sd_models.checkpoints_list.copy()
        for name, checkpoint in checkpoints.items():
            try:
                exists = os.path.exists(checkpoint.filename)
                record = {
                    "type": 'Model',
                    "name": checkpoint.name,
                    "title": checkpoint.title,
                    "filename": checkpoint.filename,
                    "hash": checkpoint.shorthash,
                    "search_term": self.search_terms_from_path(checkpoint.title),
                    "preview": self.find_preview(checkpoint.filename),
                    "local_preview": f"{os.path.splitext(checkpoint.filename)[0]}.{shared.opts.samples_format}",
                    "metadata": checkpoint.metadata,
                    "onclick": '"' + html.escape(f"""return selectCheckpoint({json.dumps(name)})""") + '"',
                    "mtime": os.path.getmtime(checkpoint.filename) if exists else 0,
                    "size": os.path.getsize(checkpoint.filename) if exists else 0,
                }
                record["info"] = self.find_info(checkpoint.filename)
                record["description"] = self.find_description(checkpoint.filename, record["info"])
                yield record
            except Exception as e:
                shared.log.debug(f"Extra networks error: type=model file={name} {e}")
        for record in self.list_reference():
            yield record

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.ckpt_dir, shared.opts.diffusers_dir, reference_dir, sd_models.model_path] if v is not None]
