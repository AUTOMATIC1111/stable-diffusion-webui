from transformers import BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers import XLMRobertaModel,XLMRobertaTokenizer
from typing import Optional

class BertSeriesConfig(BertConfig):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type="absolute", use_cache=True, classifier_dropout=None,project_dim=512, pooler_fn="average",learn_encoder=False,model_type='bert',**kwargs):

        super().__init__(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps, pad_token_id, position_embedding_type, use_cache, classifier_dropout, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        self.learn_encoder = learn_encoder

class RobertaSeriesConfig(XLMRobertaConfig):
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2,project_dim=512,pooler_fn='cls',learn_encoder=False, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        self.learn_encoder = learn_encoder


class BertSeriesModelWithTransformation(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    config_class = BertSeriesConfig

    def __init__(self, config=None, **kargs):
        # modify initialization for autoloading
        if config is None:
            config = XLMRobertaConfig()
            config.attention_probs_dropout_prob= 0.1
            config.bos_token_id=0
            config.eos_token_id=2
            config.hidden_act='gelu'
            config.hidden_dropout_prob=0.1
            config.hidden_size=1024
            config.initializer_range=0.02
            config.intermediate_size=4096
            config.layer_norm_eps=1e-05
            config.max_position_embeddings=514

            config.num_attention_heads=16
            config.num_hidden_layers=24
            config.output_past=True
            config.pad_token_id=1
            config.position_embedding_type= "absolute"

            config.type_vocab_size= 1
            config.use_cache=True
            config.vocab_size= 250002
            config.project_dim = 768
            config.learn_encoder = False
        super().__init__(config)
        self.roberta = XLMRobertaModel(config)
        self.transformation = nn.Linear(config.hidden_size,config.project_dim)
        self.pre_LN=nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        self.pooler = lambda x: x[:,0]
        self.post_init()

    def encode(self,c):
        device = next(self.parameters()).device
        text = self.tokenizer(c,
                        truncation=True,
                        max_length=77,
                        return_length=False,
                        return_overflowing_tokens=False,
                        padding="max_length",
                        return_tensors="pt")
        text["input_ids"] = torch.tensor(text["input_ids"]).to(device)
        text["attention_mask"] = torch.tensor(
            text['attention_mask']).to(device)
        features = self(**text)
        return features['projection_state']

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) :
        r"""
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # last module outputs
        sequence_output = outputs[0]


        # project every module
        sequence_output_ln = self.pre_LN(sequence_output)

        # pooler
        pooler_output = self.pooler(sequence_output_ln)
        pooler_output = self.transformation(pooler_output)
        projection_state = self.transformation(outputs.last_hidden_state)

        return {
            'pooler_output':pooler_output,
            'last_hidden_state':outputs.last_hidden_state,
            'hidden_states':outputs.hidden_states,
            'attentions':outputs.attentions,
            'projection_state':projection_state,
            'sequence_out': sequence_output
        }


class RobertaSeriesModelWithTransformation(BertSeriesModelWithTransformation):
    base_model_prefix = 'roberta'
    config_class= RobertaSeriesConfig
