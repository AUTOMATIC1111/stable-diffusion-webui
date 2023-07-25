import os

from modules import ui_extra_networks, sd_hijack, shared
from modules.ui_extra_networks import quote_js


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Textual Inversion')
        self.allow_negative_prompt = True

    def refresh(self):
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    def create_item(self, name, index=None):
        embedding = sd_hijack.model_hijack.embedding_db.word_embeddings.get(name)

        path, ext = os.path.splitext(embedding.filename)
        return {
            "name": name,
            "filename": embedding.filename,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(embedding.filename),
            "prompt": quote_js(embedding.name),
            "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            "sort_keys": {'default': index, **self.get_sort_keys(embedding.filename)},
        }

    def list_items(self):
        for index, name in enumerate(sd_hijack.model_hijack.embedding_db.word_embeddings):
            yield self.create_item(name, index)

    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
