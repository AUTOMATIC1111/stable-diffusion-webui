import os
import html
import json
import concurrent
from modules import shared, ui_extra_networks, sd_models


reference_dir = os.path.join('models', 'Reference')

class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Model')

    def refresh(self):
        shared.refresh_checkpoints()

    def list_reference(self): # pylint: disable=inconsistent-return-statements
        reference_models = shared.readfile(os.path.join('html', 'reference.json'))
        for k, v in reference_models.items():
            if shared.backend != shared.Backend.DIFFUSERS:
                if not v.get('original', False):
                    continue
                url = v.get('alt', None) or v['path']
            else:
                url = v['path']
            name = os.path.join(reference_dir, k)
            preview = v.get('preview', v['path'])
            yield {
                "type": 'Model',
                "name": name,
                "title": name,
                "filename": url,
                "preview": self.find_preview(os.path.join(reference_dir, preview)),
                "local_preview": self.find_preview_file(os.path.join(reference_dir, preview)),
                "onclick": '"' + html.escape(f"""return selectReference({json.dumps(url)})""") + '"',
                "hash": None,
                "mtime": 0,
                "size": 0,
                "info": {},
                "metadata": {},
                "description": v.get('desc', ''),
            }

    def create_item(self, name):
        record = None
        try:
            checkpoint: sd_models.CheckpointInfo = sd_models.checkpoints_list.get(name)
            exists = os.path.exists(checkpoint.filename)
            record = {
                "type": 'Model',
                "name": checkpoint.name,
                "title": checkpoint.title,
                "filename": checkpoint.filename,
                "hash": checkpoint.shorthash,
                "preview": self.find_preview(checkpoint.filename),
                "local_preview": f"{os.path.splitext(checkpoint.filename)[0]}.{shared.opts.samples_format}",
                "metadata": checkpoint.metadata,
                "onclick": '"' + html.escape(f"""return selectCheckpoint({json.dumps(name)})""") + '"',
                "mtime": os.path.getmtime(checkpoint.filename) if exists else 0,
                "size": os.path.getsize(checkpoint.filename) if exists else 0,
            }
            record["info"] = self.find_info(checkpoint.filename)
            record["description"] = self.find_description(checkpoint.filename, record["info"])
        except Exception as e:
            shared.log.debug(f"Extra networks error: type=model file={name} {e}")
        return record

    def list_items(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
            future_items = {executor.submit(self.create_item, cp): cp for cp in list(sd_models.checkpoints_list.copy())}
            for future in concurrent.futures.as_completed(future_items):
                item = future.result()
                if item is not None:
                    yield item
        for record in self.list_reference():
            yield record

    def allowed_directories_for_previews(self):
        if shared.backend == shared.Backend.DIFFUSERS:
            return [v for v in [shared.opts.ckpt_dir, shared.opts.diffusers_dir, reference_dir] if v is not None]
        else:
            return [v for v in [shared.opts.ckpt_dir, reference_dir, sd_models.model_path] if v is not None]
