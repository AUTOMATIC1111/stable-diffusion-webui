# MBartTranslator :
# Author : ParisNeo
# Description : This script translates Stable diffusion prompt from one of the 50 languages supported by MBART
#    It uses MBartTranslator class that provides a simple interface for translating text using the MBart language model.


from collections import namedtuple
from copy import copy
from itertools import permutations, chain
import random
import csv
from io import StringIO
from PIL import Image
import numpy as np

import modules.scripts as scripts
import gradio as gr

from modules import images, paths, sd_samplers, processing, sd_models, sd_vae
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae
import glob
import os
import re

from modules.ui_components import ToolButton

from transformers import MBart50TokenizerFast, MBartForConditionalGeneration


class MBartTranslator:
    """MBartTranslator class provides a simple interface for translating text using the MBart language model.

    The class can translate between 50 languages and is based on the "facebook/mbart-large-50-many-to-one-mmt"
    pre-trained MBart model. However, it is possible to use a different MBart model by specifying its name.

    Attributes:
        model (MBartForConditionalGeneration): The MBart language model.
        tokenizer (MBart50TokenizerFast): The MBart tokenizer.
    """

    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt", src_lang=None, tgt_lang=None):

        self.supported_languages = [
            "ar_AR",
            "de_DE",
            "en_XX",
            "es_XX",
            "fr_XX",
            "hi_IN",
            "it_IT",
            "ja_XX",
            "ko_XX",
            "pt_XX",
            "ru_XX",
            "zh_XX",
            "af_ZA",
            "bn_BD",
            "bs_XX",
            "ca_XX",
            "cs_CZ",
            "da_XX",
            "el_GR",
            "et_EE",
            "fa_IR",
            "fi_FI",
            "gu_IN",
            "he_IL",
            "hi_XX",
            "hr_HR",
            "hu_HU",
            "id_ID",
            "is_IS",
            "ja_XX",
            "jv_XX",
            "ka_GE",
            "kk_XX",
            "km_KH",
            "kn_IN",
            "ko_KR",
            "lo_LA",
            "lt_LT",
            "lv_LV",
            "mk_MK",
            "ml_IN",
            "mr_IN",
            "ms_MY",
            "ne_NP",
            "nl_XX",
            "no_XX",
            "pl_XX",
            "ro_RO",
            "si_LK",
            "sk_SK",
            "sl_SI",
            "sq_AL",
            "sr_XX",
            "sv_XX",
            "sw_TZ",
            "ta_IN",
            "te_IN",
            "th_TH",
            "tl_PH",
            "tr_TR",
            "uk_UA",
            "ur_PK",
            "vi_VN",
            "war_PH",
            "yue_XX",
            "zh_CN",
            "zh_TW",
        ]
        print("Building translator")
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
        print("Translator ready")

    def translate(self, text: str, input_language: str, output_language: str) -> str:
        """Translate the given text from the input language to the output language.

        Args:
            text (str): The text to translate.
            input_language (str): The input language code (e.g. "hi_IN" for Hindi).
            output_language (str): The output language code (e.g. "en_US" for English).

        Returns:
            str: The translated text.
        """
        if input_language not in self.supported_languages:
            raise ValueError(f"Input language not supported. Supported languages: {self.supported_languages}")
        if output_language not in self.supported_languages:
            raise ValueError(f"Output language not supported. Supported languages: {self.supported_languages}")

        self.tokenizer.src_lang = input_language
        encoded_input = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_input, forced_bos_token_id=self.tokenizer.lang_code_to_id[output_language]
        )
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return translated_text[0]



class AxisOption:
    def __init__(self, label, language_code):
        self.label = label
        self.language_code = language_code



language_options = [
    AxisOption("عربية", "ar_AR"),
    AxisOption("Deutsch", "de_DE"),
    AxisOption("Español", "es_XX"),
    AxisOption("Français", "fr_XX"),
    AxisOption("हिन्दी", "hi_IN"),
    AxisOption("Italiano", "it_IT"),
    AxisOption("日本語", "ja_XX"),
    AxisOption("한국어", "ko_XX"),
    AxisOption("Português", "pt_XX"),
    AxisOption("Русский", "ru_XX"),
    AxisOption("中文", "zh_XX"),
    AxisOption("Afrikaans", "af_ZA"),
    AxisOption("বাংলা", "bn_BD"),
    AxisOption("Bosanski", "bs_XX"),
    AxisOption("Català", "ca_XX"),
    AxisOption("Čeština", "cs_CZ"),
    AxisOption("Dansk", "da_XX"),
    AxisOption("Ελληνικά", "el_GR"),
    AxisOption("Eesti", "et_EE"),
    AxisOption("فارسی", "fa_IR"),
    AxisOption("Suomi", "fi_FI"),
    AxisOption("ગુજરાતી", "gu_IN"),
    AxisOption("עברית", "he_IL"),
    AxisOption("हिन्दी", "hi_XX"),
    AxisOption("Hrvatski", "hr_HR"),
    AxisOption("Magyar", "hu_HU"),
    AxisOption("Bahasa Indonesia", "id_ID"),
    AxisOption("Íslenska", "is_IS"),
    AxisOption("日本語", "ja_XX"),
    AxisOption("Javanese", "jv_XX"),
    AxisOption("ქართული", "ka_GE"),
    AxisOption("Қазақ", "kk_XX"),
    AxisOption("ខ្មែរ", "km_KH"),
    AxisOption("ಕನ್ನಡ", "kn_IN"),
    AxisOption("한국어", "ko_KR"),
    AxisOption("ລາວ", "lo_LA"),
    AxisOption("Lietuvių", "lt_LT"),
    AxisOption("Latviešu", "lv_LV"),
    AxisOption("Македонски", "mk_MK"),
    AxisOption("മലയാളം", "ml_IN"),
    AxisOption("मराठी", "mr_IN"),
    AxisOption("Bahasa Melayu", "ms_MY"),
    AxisOption("नेपाली", "ne_NP"),
    AxisOption("Nederlands", "nl_XX"),
    AxisOption("Norsk", "no_XX"),
    AxisOption("Polski", "pl_XX"),
    AxisOption("Română", "ro_RO"),
    AxisOption("සිංහල", "si_LK"),
    AxisOption("Slovenčina", "sk_SK"),
    AxisOption("Slovenščina", "sl_SI"),
    AxisOption("Shqip", "sq_AL"),    
]



class SharedSettingsStackHelper(object):
    def __enter__(self):
        self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        self.vae = opts.sd_vae
  
    def __exit__(self, exc_type, exc_value, tb):
        opts.data["sd_vae"] = self.vae
        modules.sd_models.reload_model_weights()
        modules.sd_vae.reload_vae_weights()

        opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers



class Script(scripts.Script):
    def title(self):
        return "Prompt Translate to english"

    def ui(self, is_img2img):
        self.current_axis_options = [x for x in language_options]

        with gr.Row():
            with gr.Column(scale=19):
                with gr.Row():
                    language = gr.Dropdown(label="Source language", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[1].label, type="index", elem_id=self.elem_id("x_type"))

        return [language]

    def run(self, p, language):
        if not hasattr(self, "translator"):
            self.translator = MBartTranslator()
        print(f"Translating to English from {language_options[language].label}")
        print(f"Initial prompt:{p.prompt}")
        ln_code = language_options[language].language_code
        p.prompt = self.translator.translate(p.prompt, ln_code, "en_XX")
        print(f"Translated prompt:{p.prompt}")
        if not opts.return_grid:
            p.batch_size = 1


        with SharedSettingsStackHelper():
            if shared.state.interrupted:
                return Processed(p, [], p.seed, "")

            pc = copy(p)
            processed = process_images(pc)


        if opts.grid_save:
            grid_infotext = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)
            images.save_image(processed.images[0], p.outpath_grids, "xyz_grid", info=grid_infotext[0], extension=opts.grid_format, prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        return processed
