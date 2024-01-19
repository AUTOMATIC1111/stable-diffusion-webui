import json
import os
from modules import shared, sd_hijack, sd_models, ui_extra_networks, files_cache
from modules.textual_inversion.textual_inversion import Embedding


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Embedding')
        self.allow_negative_prompt = True
        self.embeddings = []

    def refresh(self):
        if sd_models.model_data.sd_model is None:
            return
        if shared.backend == shared.Backend.ORIGINAL:
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        elif hasattr(sd_models.model_data.sd_model, 'embedding_db'):
            sd_models.model_data.sd_model.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    def create_item(self, embedding: Embedding):
        record = None
        try:
            tags = {}
            if embedding.tag is not None:
                tags[embedding.tag]=1
            name = os.path.splitext(embedding.basename)[0]
            record = {
                "type": 'Embedding',
                "name": name,
                "filename": embedding.filename,
                "prompt": json.dumps(f" {os.path.splitext(embedding.name)[0]}"),
                "tags": tags,
                "mtime": os.path.getmtime(embedding.filename),
                "size": os.path.getsize(embedding.filename),
            }
            record["info"] = self.find_info(embedding.filename)
            record["description"] = self.find_description(embedding.filename, record["info"])
        except Exception as e:
            shared.log.debug(f"Extra networks error: type=embedding file={embedding.filename} {e}")
        return record

    def list_items(self):
        if sd_models.model_data.sd_model is None:
            candidates = list(files_cache.list_files(shared.opts.embeddings_dir, ext_filter=['.pt', '.safetensors'], recursive=files_cache.not_hidden))
            self.embeddings = [
                Embedding(vec=0, name=os.path.basename(embedding_path), filename=embedding_path)
                for embedding_path
                in candidates
            ]
        elif shared.backend == shared.Backend.ORIGINAL:
            self.embeddings = list(sd_hijack.model_hijack.embedding_db.word_embeddings.values())
        elif hasattr(sd_models.model_data.sd_model, 'embedding_db'):
            self.embeddings = list(sd_models.model_data.sd_model.embedding_db.word_embeddings.values())
        else:
            self.embeddings = []
        self.embeddings = sorted(self.embeddings, key=lambda emb: emb.filename)

        items = [self.create_item(embedding) for embedding in self.embeddings]
        self.update_all_previews(items)
        return items

    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
