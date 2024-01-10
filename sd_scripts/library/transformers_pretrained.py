from transformers import PreTrainedModel, PreTrainedTokenizerBase

print(f"save PreTrainedTokenizerBase.from_pretrained: {PreTrainedTokenizerBase.from_pretrained}")
# 记录原本的from_pretrained语义
ori_tokenizer_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__
ori_model_from_pretrained = PreTrainedModel.from_pretrained.__func__


print(f"save PreTrainedTokenizerBase.from_pretrained hash:{hash(ori_tokenizer_from_pretrained)}")
