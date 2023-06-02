from modules import sd_hijack_clip
from modules import shared


def process_text_old(self: sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase, texts):
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

                mult_change = self.token_mults.get(token) if shared.opts.enable_emphasis else None
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


def forward_old(self: sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase, texts):
    batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = process_text_old(self, texts)

    self.hijack.comments += hijack_comments

    if used_custom_terms:
        embedding_names = ", ".join(f"{word} [{checksum}]" for word, checksum in used_custom_terms)
        self.hijack.comments.append(f"Used embeddings: {embedding_names}")

    self.hijack.fixes = hijack_fixes
    return self.process_tokens(remade_batch_tokens, batch_multipliers)
