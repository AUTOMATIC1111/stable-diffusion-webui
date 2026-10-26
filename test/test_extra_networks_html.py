from html.parser import HTMLParser
from types import SimpleNamespace

import pytest


class CardFieldsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.search_terms = []
        self.in_span = False

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))
        if tag == "span":
            self.search_terms.append("")
            self.in_span = True

    def handle_endtag(self, tag):
        if tag == "span":
            self.in_span = False

    def handle_data(self, data):
        if self.in_span:
            self.search_terms[-1] += data


@pytest.mark.parametrize("filename", [
    "models/Lora/ordinary.safetensors",
    "models/Lora/cat & dog.safetensors",
    'models/Lora/a" data-extra="marker.safetensors',
    "models/Lora/<b>marker</b>.safetensors",
    'models/Lora/a"><b>marker</b>.safetensors',
    "models/Lora/l'art &amp; café.safetensors",
])
@pytest.mark.parametrize("use_template", [False, True])
@pytest.mark.parametrize("field", ["filename", "search_terms"])
def test_card_filename_and_search_terms_remain_text(initialize, monkeypatch, filename, use_template, field):
    from modules import shared, ui_extra_networks

    monkeypatch.setattr(shared, "opts", SimpleNamespace(
        extra_networks_card_height=0,
        extra_networks_card_width=0,
        extra_networks_card_text_scale=1,
        extra_networks_hidden_models="Always",
        extra_networks_card_show_desc=False,
        extra_networks_card_description_is_html=False,
    ))
    page = ui_extra_networks.ExtraNetworksPage("Lora")
    monkeypatch.setattr(page, "allowed_directories_for_previews", lambda: [])
    search_terms = [filename, "alias"] if field == "search_terms" else ["model", "alias"]
    filename = filename if field == "filename" else "models/Lora/ordinary.safetensors"
    item = {
        "filename": filename,
        "name": "model",
        "prompt": "''",
        "local_preview": "preview.png",
        "search_terms": search_terms,
    }
    template = "{copy_path_button}{search_terms}" if use_template else None
    rendered = page.create_item_html("txt2img", item, template)
    if not use_template:
        rendered = rendered["copy_path_button"] + rendered["search_terms"]

    parser = CardFieldsParser()
    parser.feed(rendered)
    assert [tag for tag, attrs in parser.elements] == ["div", "span", "span"]
    assert parser.elements[0][1] == {
        "class": "copy-path-button card-button",
        "title": "Copy path to clipboard",
        "onclick": "extraNetworksCopyCardPath(event)",
        "data-clipboard-text": filename,
    }
    assert parser.search_terms == search_terms
    assert item["filename"] == filename
    assert item["search_terms"] == search_terms
