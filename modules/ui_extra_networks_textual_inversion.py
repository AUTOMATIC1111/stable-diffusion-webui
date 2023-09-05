import json
import os

from modules import shared, sd_hijack, sd_models, ui_extra_networks
from modules.textual_inversion.textual_inversion import Embedding


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Textual Inversion')
        self.allow_negative_prompt = True

    def refresh(self):
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    def list_items(self):
        if sd_models.model_data.sd_model is None:
            embeddings = []
            for root, _dirs, fns in os.walk(shared.opts.embeddings_dir, followlinks=True):
                for fn in fns:
                    if fn.lower().endswith(".pt") or fn.lower().endswith(".safetensors"):
                        embedding = Embedding(0, fn)
                        embedding.filename = os.path.join(root, fn)
                        embeddings.append(embedding)
        elif shared.backend == shared.Backend.ORIGINAL:
            embeddings = list(sd_hijack.model_hijack.embedding_db.word_embeddings.values())
        elif hasattr(sd_models.model_data.sd_model, 'embedding_db'):
            embeddings = list(sd_models.model_data.sd_model.embedding_db.word_embeddings.values())
        else:
            embeddings = []
        for embedding in embeddings:
            path, _ext = os.path.splitext(embedding.filename)
            yield {
                "name": os.path.splitext(embedding.name)[0],
                "filename": embedding.filename,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "info": self.find_info(path),
                "search_term": self.search_terms_from_path(embedding.filename),
                "prompt": json.dumps(os.path.splitext(embedding.name)[0]),
                "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            }

    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
