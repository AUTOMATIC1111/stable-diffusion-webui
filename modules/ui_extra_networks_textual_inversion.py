import json
import os
import concurrent
from modules import shared, sd_hijack, sd_models, ui_extra_networks
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
            path, _ext = os.path.splitext(embedding.filename)
            tags = {}
            if embedding.tag is not None:
                tags[embedding.tag]=1
            name = os.path.splitext(embedding.basename)[0]
            record = {
                "type": 'Embedding',
                "name": name,
                "filename": embedding.filename,
                "preview": self.find_preview(embedding.filename),
                "prompt": json.dumps(f" {os.path.splitext(embedding.name)[0]}"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
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

        def list_folder(folder):
            for filename in os.listdir(folder):
                fn = os.path.join(folder, filename)
                if os.path.isfile(fn) and (fn.lower().endswith(".pt") or fn.lower().endswith(".safetensors")):
                    embedding = Embedding(vec=0, name=os.path.basename(fn), filename=fn)
                    embedding.filename = fn
                    self.embeddings.append(embedding)
                elif os.path.isdir(fn) and not fn.startswith('.'):
                    list_folder(fn)

        if sd_models.model_data.sd_model is None:
            self.embeddings = []
            list_folder(shared.opts.embeddings_dir)
        elif shared.backend == shared.Backend.ORIGINAL:
            self.embeddings = list(sd_hijack.model_hijack.embedding_db.word_embeddings.values())
        elif hasattr(sd_models.model_data.sd_model, 'embedding_db'):
            self.embeddings = list(sd_models.model_data.sd_model.embedding_db.word_embeddings.values())
        else:
            self.embeddings = []
        self.embeddings = sorted(self.embeddings, key=lambda emb: emb.filename)

        with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
            future_items = {executor.submit(self.create_item, net): net for net in self.embeddings}
            for future in concurrent.futures.as_completed(future_items):
                item = future.result()
                if item is not None:
                    yield item

    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
