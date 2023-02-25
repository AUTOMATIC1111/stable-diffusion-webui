import json
import os

from modules import ui_extra_networks, sd_hijack


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Textual Inversion')
        self.allow_negative_prompt = True

    def refresh(self):
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    def list_items(self):
        for embedding in sd_hijack.model_hijack.embedding_db.word_embeddings.values():
            path, ext = os.path.splitext(embedding.filename)
            previews = [path + ".png", path + ".preview.png"]

            preview = None
            for file in previews:
                if os.path.isfile(file):
                    preview = self.link_preview(file)
                    break

            yield {
                "name": embedding.name,
                "filename": embedding.filename,
                "preview": preview,
                "search_term": self.search_terms_from_path(embedding.filename),
                "prompt": json.dumps(embedding.name),
                "local_preview": path + ".preview.png",
            }

    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
