from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from diffusers.loaders.textual_inversion import (
    TextualInversionLoaderMixin,
    load_textual_inversion_state_dicts,
    logger,
    nn,
)

from modules import shared
from modules.patches import patch_method

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
except Exception:
    pass

@patch_method(TextualInversionLoaderMixin)
def load_textual_inversion(
    self: TextualInversionLoaderMixin,
    pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    token: Optional[Union[str, List[str]]] = None,
    tokenizer: Optional["PreTrainedTokenizer"] = None,
    text_encoder: Optional["PreTrainedModel"] = None,
    **kwargs, # pylint: disable=W0613
):

    # 1. Set correct tokenizer and text encoder
    tokenizer: PreTrainedTokenizer = tokenizer or getattr(self, "tokenizer", None)
    text_encoder: PreTrainedModel  = text_encoder or getattr(self, "text_encoder", None)
    loaded_model_names_or_paths = {}

    assert tokenizer and text_encoder, 'Can not resolve `tokenizer` or `text_encoder`'

    # 2. Normalize inputs
    pretrained_model_name_or_paths = (
        [pretrained_model_name_or_path]
        if not isinstance(pretrained_model_name_or_path, list)
        else pretrained_model_name_or_path
    )
    tokens = len(pretrained_model_name_or_paths) * [token] if (isinstance(token, str) or token is None) else token
    assert len(tokens) == len(pretrained_model_name_or_paths), f'Number of Models ({len(pretrained_model_name_or_paths)}) and Tokens ({len(tokens)}) must be equal.'
    number_of_models = len(pretrained_model_name_or_paths)
    token_data = {}

    # Build a unique list of Shape-Token/Embedding-Shape pairs, mapped with an associated `name_or_path` for reporting
    expected_emb_dim = text_encoder.get_input_embeddings().weight.shape[-1]
    for idx in range(number_of_models):
        try:
            name_or_path = pretrained_model_name_or_paths[idx]
            token = tokens[idx]
            _embedding_data = {
                shape_token: (embedding_shape, name_or_path )
                for shape_token, embedding_shape in zip(
                    # 5. Extend tokens and embeddings for multi vector
                    *self._extend_tokens_and_embeddings( # pylint: disable=W0212
                        # 4. Retrieve tokens and embeddings
                        *TextualInversionLoaderMixin._retrieve_tokens_and_embeddings( # pylint: disable=W0212
                            [token],
                            # 3. Load state dicts of textual embeddings
                            load_textual_inversion_state_dicts(
                                [name_or_path],
                                cache_dir=shared.opts.diffusers_dir,
                                local_files_only=True
                            ),
                            tokenizer
                        ),
                        tokenizer
                    )
                )
            }
            # 6. Make sure all embeddings have the correct size
            for embedding_shape, _ in _embedding_data.values():
                if expected_emb_dim != embedding_shape.shape[-1]:
                    #debug.error(f'Incorrect Shape: {embedding_shape.shape[-1]} vs {expected_emb_dim}')
                    raise ValueError(
                        "Loaded embeddings are of incorrect shape. Expected each textual inversion embedding "
                        "to be of shape {embedding_shape.shape[-1]}, but are {embeddings.shape[-1]} "
                    )

            token_data.update(_embedding_data)
        except Exception:
            if number_of_models == 1:
                raise

    # 7. Now we can be sure that loading the embedding matrix works
    # < Unsafe code:

    # 7.1 Offload all hooks in case the pipeline was cpu offloaded before make sure, we offload and onload again
    is_model_cpu_offload = False
    is_sequential_cpu_offload = False

    for _, component in self.components.items():
        if isinstance(component, nn.Module):
            if hasattr(component, "_hf_hook"):
                is_model_cpu_offload = isinstance(getattr(component, "_hf_hook"), CpuOffload)  # noqa: B009
                is_sequential_cpu_offload = isinstance(getattr(component, "_hf_hook"), AlignDevicesHook)  # noqa: B009
                logger.info(
                    "Accelerate hooks detected. Since you have called `load_textual_inversion()`, the previous hooks will be first removed. Then the textual inversion parameters will be loaded and the hooks will be applied again."
                )
                remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

    # 7.2 save expected device and dtype
    device = text_encoder.device
    dtype = text_encoder.dtype

    # 7.2 Add Tokens to the Tokenizer
    tokens_to_add = list(token_data)
    tokenizer_size = len(tokenizer)

    # 7.3 Increase token embedding matrix
    text_encoder.resize_token_embeddings(tokenizer_size)

    input_embeddings = text_encoder.get_input_embeddings().weight

    unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # 7.4 Load token and embedding
    for token_id, load_token in zip(tokenizer.convert_tokens_to_ids(tokens_to_add), tokens_to_add):
        if token_id <= unk_token_id:
            raise RuntimeError(f'Processed Shape-Token `{load_token}` does not resolve to a new Token ID')
        embedding = token_data[load_token][0]
        path = token_data[load_token][1]
        input_embeddings.data[token_id] = embedding
        loaded_model_names_or_paths[path] = True

    input_embeddings.to(dtype=dtype, device=device)

    # 7.5 Offload the model again
    if is_model_cpu_offload:
        self.enable_model_cpu_offload()
    elif is_sequential_cpu_offload:
        self.enable_sequential_cpu_offload()

    return list(loaded_model_names_or_paths) if number_of_models != 1 else None
    # / Unsafe Code >
