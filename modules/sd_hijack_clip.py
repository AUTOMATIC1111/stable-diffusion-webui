import math

import torch

from modules import prompt_parser, devices
from modules.shared import opts


def get_target_prompt_token_count(token_count):
    return math.ceil(max(token_count, 1) / 75) * 75


class FrozenCLIPEmbedderWithCustomWordsBase(torch.nn.Module):
    def __init__(self, wrapped, hijack):
        super().__init__()
        self.wrapped = wrapped
        self.hijack = hijack

    def tokenize(self, texts):
        raise NotImplementedError

    def encode_with_transformers(self, tokens):
        raise NotImplementedError

    def encode_embedding_init_text(self, init_text, nvpt):
        raise NotImplementedError

    def tokenize_line(self, line, used_custom_terms, hijack_comments):
        if opts.enable_emphasis:
            parsed = prompt_parser.parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenize([text for text, _ in parsed])

        fixes = []
        remade_tokens = []
        multipliers = []
        last_comma = -1

        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]

                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, i)

                if token == self.comma_token:
                    last_comma = len(remade_tokens)
                elif opts.comma_padding_backtrack != 0 and max(len(remade_tokens), 1) % 75 == 0 and last_comma != -1 and len(remade_tokens) - last_comma <= opts.comma_padding_backtrack:
                    last_comma += 1
                    reloc_tokens = remade_tokens[last_comma:]
                    reloc_mults = multipliers[last_comma:]

                    remade_tokens = remade_tokens[:last_comma]
                    length = len(remade_tokens)

                    rem = int(math.ceil(length / 75)) * 75 - length
                    remade_tokens += [self.id_end] * rem + reloc_tokens
                    multipliers = multipliers[:last_comma] + [1.0] * rem + reloc_mults

                if embedding is None:
                    remade_tokens.append(token)
                    multipliers.append(weight)
                    i += 1
                else:
                    emb_len = int(embedding.vec.shape[0])
                    iteration = len(remade_tokens) // 75
                    if (len(remade_tokens) + emb_len) // 75 != iteration:
                        rem = (75 * (iteration + 1) - len(remade_tokens))
                        remade_tokens += [self.id_end] * rem
                        multipliers += [1.0] * rem
                        iteration += 1
                    fixes.append((iteration, (len(remade_tokens) % 75, embedding)))
                    remade_tokens += [0] * emb_len
                    multipliers += [weight] * emb_len
                    used_custom_terms.append((embedding.name, embedding.checksum()))
                    i += embedding_length_in_tokens

        token_count = len(remade_tokens)
        prompt_target_length = get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)

        remade_tokens = remade_tokens + [self.id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add

        return remade_tokens, fixes, multipliers, token_count

    def process_text(self, texts):
        used_custom_terms = []
        remade_batch_tokens = []
        hijack_comments = []
        hijack_fixes = []
        token_count = 0

        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, fixes, multipliers = cache[line]
            else:
                remade_tokens, fixes, multipliers, current_token_count = self.tokenize_line(line, used_custom_terms, hijack_comments)
                token_count = max(current_token_count, token_count)

                cache[line] = (remade_tokens, fixes, multipliers)

            remade_batch_tokens.append(remade_tokens)
            hijack_fixes.append(fixes)
            batch_multipliers.append(multipliers)

        return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

    def process_text_old(self, texts):
        id_start = self.id_start
        id_end = self.id_end
        maxlen = self.wrapped.max_length  # you get to stay at 77
        used_custom_terms = []
        remade_batch_tokens = []
        hijack_comments = []
        hijack_fixes = []
        token_count = 0

        cache = {}
        batch_tokens = self.tokenize(texts)
        batch_multipliers = []
        for tokens in batch_tokens:
            tuple_tokens = tuple(tokens)

            if tuple_tokens in cache:
                remade_tokens, fixes, multipliers = cache[tuple_tokens]
            else:
                fixes = []
                remade_tokens = []
                multipliers = []
                mult = 1.0

                i = 0
                while i < len(tokens):
                    token = tokens[i]

                    embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, i)

                    mult_change = self.token_mults.get(token) if opts.enable_emphasis else None
                    if mult_change is not None:
                        mult *= mult_change
                        i += 1
                    elif embedding is None:
                        remade_tokens.append(token)
                        multipliers.append(mult)
                        i += 1
                    else:
                        emb_len = int(embedding.vec.shape[0])
                        fixes.append((len(remade_tokens), embedding))
                        remade_tokens += [0] * emb_len
                        multipliers += [mult] * emb_len
                        used_custom_terms.append((embedding.name, embedding.checksum()))
                        i += embedding_length_in_tokens

                if len(remade_tokens) > maxlen - 2:
                    vocab = {v: k for k, v in self.wrapped.tokenizer.get_vocab().items()}
                    ovf = remade_tokens[maxlen - 2:]
                    overflowing_words = [vocab.get(int(x), "") for x in ovf]
                    overflowing_text = self.wrapped.tokenizer.convert_tokens_to_string(''.join(overflowing_words))
                    hijack_comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

                token_count = len(remade_tokens)
                remade_tokens = remade_tokens + [id_end] * (maxlen - 2 - len(remade_tokens))
                remade_tokens = [id_start] + remade_tokens[0:maxlen - 2] + [id_end]
                cache[tuple_tokens] = (remade_tokens, fixes, multipliers)

            multipliers = multipliers + [1.0] * (maxlen - 2 - len(multipliers))
            multipliers = [1.0] + multipliers[0:maxlen - 2] + [1.0]

            remade_batch_tokens.append(remade_tokens)
            hijack_fixes.append(fixes)
            batch_multipliers.append(multipliers)
        return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

    def forward(self, text):
        use_old = opts.use_old_emphasis_implementation
        if use_old:
            batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = self.process_text_old(text)
        else:
            batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = self.process_text(text)

        self.hijack.comments += hijack_comments

        if len(used_custom_terms) > 0:
            self.hijack.comments.append("Used embeddings: " + ", ".join([f'{word} [{checksum}]' for word, checksum in used_custom_terms]))

        if use_old:
            self.hijack.fixes = hijack_fixes
            return self.process_tokens(remade_batch_tokens, batch_multipliers)

        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]
            rem_multipliers = [x[75:] for x in batch_multipliers]

            self.hijack.fixes = []
            for unfiltered in hijack_fixes:
                fixes = []
                for fix in unfiltered:
                    if fix[0] == i:
                        fixes.append(fix[1])
                self.hijack.fixes.append(fixes)

            tokens = []
            multipliers = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:75])
                    multipliers.append(batch_multipliers[j][:75])
                else:
                    tokens.append([self.id_end] * 75)
                    multipliers.append([1.0] * 75)

            z1 = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), axis=-2)

            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1

        return z

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        if not opts.use_old_emphasis_implementation:
            remade_batch_tokens = [[self.id_start] + x[:75] + [self.id_end] for x in remade_batch_tokens]
            batch_multipliers = [[1.0] + x[:75] + [1.0] for x in batch_multipliers]

        tokens = torch.asarray(remade_batch_tokens).to(devices.device)

        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index+1:tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [x + [1.0] * (75 - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(batch_multipliers_of_same_length).to(devices.device)
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z


class FrozenCLIPEmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWordsBase):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.tokenizer = wrapped.tokenizer
        self.comma_token = [v for k, v in self.tokenizer.get_vocab().items() if k == ',</w>'][0]

        self.token_mults = {}
        tokens_with_parens = [(k, v) for k, v in self.tokenizer.get_vocab().items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

        self.id_start = self.wrapped.tokenizer.bos_token_id
        self.id_end = self.wrapped.tokenizer.eos_token_id
        self.id_pad = self.id_end

    def tokenize(self, texts):
        tokenized = self.wrapped.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)

        if opts.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.transformer.text_model.embeddings
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)

        return embedded
