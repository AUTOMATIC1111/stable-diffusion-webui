from modules import scripts, shared, script_callbacks
import re


def strip_comments(text):
    text = re.sub('(^|\n)#[^\n]*(\n|$)', '\n', text)  # while line comment
    text = re.sub('#[^\n]*(\n|$)', '\n', text)  # in the middle of the line comment

    return text


class ScriptStripComments(scripts.Script):
    def title(self):
        return "Comments"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        if not shared.opts.enable_prompt_comments:
            return

        p.all_prompts = [strip_comments(x) for x in p.all_prompts]
        p.all_negative_prompts = [strip_comments(x) for x in p.all_negative_prompts]

        p.main_prompt = strip_comments(p.main_prompt)
        p.main_negative_prompt = strip_comments(p.main_negative_prompt)


def before_token_counter(params: script_callbacks.BeforeTokenCounterParams):
    if not shared.opts.enable_prompt_comments:
        return

    params.prompt = strip_comments(params.prompt)


script_callbacks.on_before_token_counter(before_token_counter)


shared.options_templates.update(shared.options_section(('sd', "Stable Diffusion", "sd"), {
    "enable_prompt_comments": shared.OptionInfo(True, "Enable comments").info("Use # anywhere in the prompt to hide the text between # and the end of the line from the generation."),
}))
