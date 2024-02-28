from transformers import CLIPTokenizer, AutoTokenizer
from ..models import SDTextEncoder, SDXLTextEncoder, SDXLTextEncoder2
import torch, os
from modules import shared, scripts

def tokenize_long_prompt(tokenizer, prompt):
    # Get model_max_length from self.tokenizer
    length = tokenizer.model_max_length

    # To avoid the warning. set self.tokenizer.model_max_length to +oo.
    tokenizer.model_max_length = 99999999

    # Tokenize it!
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Determine the real length.
    max_length = (input_ids.shape[1] + length - 1) // length * length

    # Restore tokenizer.model_max_length
    tokenizer.model_max_length = length
    
    # Tokenize it again with fixed length.
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    ).input_ids

    # Reshape input_ids to fit the text encoder.
    num_sentence = input_ids.shape[1] // length
    input_ids = input_ids.reshape((num_sentence, length))
    
    return input_ids


class BeautifulPrompt:
    def __init__(self, tokenizer_path="configs/beautiful_prompt/tokenizer", model=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.template = 'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {raw_prompt}\nOutput:'
    
    def __call__(self, raw_prompt):
        model_input = self.template.format(raw_prompt=raw_prompt)
        input_ids = self.tokenizer.encode(model_input, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=384,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1
        )
        prompt = raw_prompt + ", " + self.tokenizer.batch_decode(
            outputs[:, input_ids.size(1):],
            skip_special_tokens=True
        )[0].strip()
        return prompt


class SDPrompter:
    def __init__(self, tokenizer_path=scripts.basedir()+"/handlers/diffstudio/configs/stable_diffusion/tokenizer"):
        # We use the tokenizer implemented by transformers
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.keyword_dict = {}
        self.beautiful_prompt: BeautifulPrompt = None
    

    def encode_prompt(self, text_encoder: SDTextEncoder, prompt, clip_skip=1, device="cuda", positive=True):
        # Textual Inversion
        for keyword in self.keyword_dict:
            if keyword in prompt:
                prompt = prompt.replace(keyword, self.keyword_dict[keyword])

        # Beautiful Prompt
        if positive and self.beautiful_prompt is not None:
            prompt = self.beautiful_prompt(prompt)
            print(f"Your prompt is refined by BeautifulPrompt: \"{prompt}\"")
        
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(device)
        prompt_emb = text_encoder(input_ids, clip_skip=clip_skip)
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))

        return prompt_emb
    
    def load_textual_inversion(self, textual_inversion_dict):
        self.keyword_dict = {}
        additional_tokens = []
        for keyword in textual_inversion_dict:
            tokens, _ = textual_inversion_dict[keyword]
            additional_tokens += tokens
            self.keyword_dict[keyword] = " " + " ".join(tokens) + " "
        self.tokenizer.add_tokens(additional_tokens)

    def load_beautiful_prompt(self, model, model_path):
        model_folder = os.path.dirname(model_path)
        self.beautiful_prompt = BeautifulPrompt(tokenizer_path=model_folder, model=model)
        if model_folder.endswith("v2"):
            self.beautiful_prompt.template = """Converts a simple image description into a prompt. \
Prompts are formatted as multiple related tags separated by commas, plus you can use () to increase the weight, [] to decrease the weight, \
or use a number to specify the weight. You should add appropriate words to make the images described in the prompt more aesthetically pleasing, \
but make sure there is a correlation between the input and output.\n\
### Input: {raw_prompt}\n### Output:"""


class SDXLPrompter:
    def __init__(
        self,
        tokenizer_path="configs/stable_diffusion/tokenizer",
        tokenizer_2_path="configs/stable_diffusion_xl/tokenizer_2"
    ):
        # We use the tokenizer implemented by transformers
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
        self.keyword_dict = {}
        self.beautiful_prompt: BeautifulPrompt = None
    
    def encode_prompt(
        self,
        text_encoder: SDXLTextEncoder,
        text_encoder_2: SDXLTextEncoder2,
        prompt,
        clip_skip=1,
        clip_skip_2=2,
        positive=True,
        device="cuda"
    ):
        # Textual Inversion
        for keyword in self.keyword_dict:
            if keyword in prompt:
                prompt = prompt.replace(keyword, self.keyword_dict[keyword])

        # Beautiful Prompt
        if positive and self.beautiful_prompt is not None:
            prompt = self.beautiful_prompt(prompt)
            print(f"Your prompt is refined by BeautifulPrompt: \"{prompt}\"")
        
        # 1
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(device)
        prompt_emb_1 = text_encoder(input_ids, clip_skip=clip_skip)

        # 2
        input_ids_2 = tokenize_long_prompt(self.tokenizer_2, prompt).to(device)
        add_text_embeds, prompt_emb_2 = text_encoder_2(input_ids_2, clip_skip=clip_skip_2)

        # Merge
        prompt_emb = torch.concatenate([prompt_emb_1, prompt_emb_2], dim=-1)

        # For very long prompt, we only use the first 77 tokens to compute `add_text_embeds`.
        add_text_embeds = add_text_embeds[0:1]
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))
        return add_text_embeds, prompt_emb
    
    def load_textual_inversion(self, textual_inversion_dict):
        self.keyword_dict = {}
        additional_tokens = []
        for keyword in textual_inversion_dict:
            tokens, _ = textual_inversion_dict[keyword]
            additional_tokens += tokens
            self.keyword_dict[keyword] = " " + " ".join(tokens) + " "
        self.tokenizer.add_tokens(additional_tokens)

    def load_beautiful_prompt(self, model, model_path):
        model_folder = os.path.dirname(model_path)
        self.beautiful_prompt = BeautifulPrompt(tokenizer_path=model_folder, model=model)
        if model_folder.endswith("v2"):
            self.beautiful_prompt.template = """Converts a simple image description into a prompt. \
Prompts are formatted as multiple related tags separated by commas, plus you can use () to increase the weight, [] to decrease the weight, \
or use a number to specify the weight. You should add appropriate words to make the images described in the prompt more aesthetically pleasing, \
but make sure there is a correlation between the input and output.\n\
### Input: {raw_prompt}\n### Output:"""
