# 关于LoRA的学习。

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（arxiv）、[LoRA](https://github.com/microsoft/LoRA)（github）这是应用于Stable Diffusion“稳定扩散”的内容。

[cloneofsimo先生的代码仓库](https://github.com/cloneofsimo/lora) 我们非常感謝您提供的参考。非常感謝。

通常情況下，LoRA只适用于Linear和Kernel大小为1x1的Conv2d，但也可以將其擴展到Kernel大小为3x3的Conv2d。

Conv2d 3x3的扩展最初是由 [cloneofsimo先生的代码仓库](https://github.com/cloneofsimo/lora) 
而KohakuBlueleaf先生在[LoCon](https://github.com/KohakuBlueleaf/LoCon)中揭示了其有效性。我们深深地感谢KohakuBlueleaf先生。

看起来即使在8GB VRAM上也可以勉强运行。

请同时查看关于[学习的通用文档](./train_README-zh.md)。
# 可学习的LoRA 类型

支持以下两种类型。以下是本仓库中自定义的名称。

1. __LoRA-LierLa__：(用于 __Li__ n __e__ a __r__  __La__ yers 的 LoRA，读作 "Liela")

    适用于 Linear 和卷积层 Conv2d 的 1x1 Kernel 的 LoRA

2. __LoRA-C3Lier__：(用于具有 3x3 Kernel 的卷积层和 __Li__ n __e__ a __r__ 层的 LoRA，读作 "Seria")

    除了第一种类型外，还适用于 3x3 Kernel 的 Conv2d 的 LoRA

与 LoRA-LierLa 相比，LoRA-C3Lier 可能会获得更高的准确性，因为它适用于更多的层。

在训练时，也可以使用 __DyLoRA__（将在后面介绍）。

## 请注意与所学模型相关的事项。

LoRA-LierLa可以用于AUTOMATIC1111先生的Web UI LoRA功能。

要使用LoRA-C3Liar并在Web UI中生成，请使用此处的[WebUI用extension](https://github.com/kohya-ss/sd-webui-additional-networks)。

在此存储库的脚本中，您还可以预先将经过训练的LoRA模型合并到Stable Diffusion模型中。

请注意，与cloneofsimo先生的存储库以及d8ahazard先生的[Stable-Diffusion-WebUI的Dreambooth扩展](https://github.com/d8ahazard/sd_dreambooth_extension)不兼容，因为它们进行了一些功能扩展（如下文所述）。

# 学习步骤

请先参考此存储库的README文件并进行环境设置。

## 准备数据

请参考 [关于准备学习数据](./train_README-zh.md)。

## 网络训练

使用`train_network.py`。

在`train_network.py`中，使用`--network_module`选项指定要训练的模块名称。对于LoRA模块，它应该是`network.lora`，请指定它。

请注意，学习率应该比通常的DreamBooth或fine tuning要高，建议指定为`1e-4`至`1e-3`左右。

以下是命令行示例。

```
accelerate launch --num_cpu_threads_per_process 1 train_network.py 
    --pretrained_model_name_or_path=<.ckpt或.safetensord或Diffusers版模型目录> 
    --dataset_config=<数据集配置的.toml文件> 
    --output_dir=<训练过程中的模型输出文件夹>  
    --output_name=<训练模型输出时的文件名> 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=400 
    --learning_rate=1e-4 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --save_every_n_epochs=1 
    --network_module=networks.lora
```

在这个命令行中，LoRA-LierLa将会被训练。

LoRA的模型将会被保存在通过`--output_dir`选项指定的文件夹中。关于其他选项和优化器等，请参阅[学习的通用文档](./train_README-zh.md)中的“常用选项”。

此外，还可以指定以下选项：

* `--network_dim`
  * 指定LoRA的RANK（例如：`--network_dim=4`）。默认值为4。数值越大表示表现力越强，但需要更多的内存和时间来训练。而且不要盲目增加此数值。
* `--network_alpha`
  * 指定用于防止下溢并稳定训练的alpha值。默认值为1。如果与`network_dim`指定相同的值，则将获得与以前版本相同的行为。
* `--persistent_data_loader_workers`
  * 在Windows环境中指定可大幅缩短epoch之间的等待时间。
* `--max_data_loader_n_workers`
  * 指定数据读取进程的数量。进程数越多，数据读取速度越快，可以更有效地利用GPU，但会占用主存。默认值为“`8`或`CPU同步执行线程数-1`的最小值”，因此如果主存不足或GPU使用率超过90％，则应将这些数字降低到约`2`或`1`。
* `--network_weights`
  * 在训练之前读取预训练的LoRA权重，并在此基础上进行进一步的训练。
* `--network_train_unet_only`
  * 仅启用与U-Net相关的LoRA模块。在类似fine tuning的学习中指定此选项可能会很有用。
* `--network_train_text_encoder_only`
  * 仅启用与Text Encoder相关的LoRA模块。可能会期望Textual Inversion效果。
* `--unet_lr`
  * 当在U-Net相关的LoRA模块中使用与常规学习率（由`--learning_rate`选项指定）不同的学习率时，应指定此选项。
* `--text_encoder_lr`
  * 当在Text Encoder相关的LoRA模块中使用与常规学习率（由`--learning_rate`选项指定）不同的学习率时，应指定此选项。可能最好将Text Encoder的学习率稍微降低（例如5e-5）。
* `--network_args`
  * 可以指定多个参数。将在下面详细说明。

当未指定`--network_train_unet_only`和`--network_train_text_encoder_only`时（默认情况），将启用Text Encoder和U-Net的两个LoRA模块。

# 其他的学习方法

## 学习 LoRA-C3Lier

请使用以下方式

```
--network_args "conv_dim=4"
```

DyLoRA是在这篇论文中提出的[DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation](​https://arxiv.org/abs/2210.07558)，
[其官方实现可在这里找到](​https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)。

根据论文，LoRA的rank并不是越高越好，而是需要根据模型、数据集、任务等因素来寻找合适的rank。使用DyLoRA，可以同时在指定的维度(rank)下学习多种rank的LoRA，从而省去了寻找最佳rank的麻烦。

本存储库的实现基于官方实现进行了自定义扩展（因此可能存在缺陷）。

### 本存储库DyLoRA的特点

DyLoRA训练后的模型文件与LoRA兼容。此外，可以从模型文件中提取多个低于指定维度(rank)的LoRA。

DyLoRA-LierLa和DyLoRA-C3Lier均可训练。

### 使用DyLoRA进行训练

请指定与DyLoRA相对应的`network.dylora`，例如 `--network_module=networks.dylora`。

此外，通过 `--network_args` 指定例如`--network_args "unit=4"`的参数。`unit`是划分rank的单位。例如，可以指定为`--network_dim=16 --network_args "unit=4"`。请将`unit`视为可以被`network_dim`整除的值（`network_dim`是`unit`的倍数）。

如果未指定`unit`，则默认为`unit=1`。

以下是示例说明。

```
--network_module=networks.dylora --network_dim=16 --network_args "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "unit=4"
```

对于DyLoRA-C3Lier，需要在 `--network_args` 中指定 `conv_dim`，例如 `conv_dim=4`。与普通的LoRA不同，`conv_dim`必须与`network_dim`具有相同的值。以下是一个示例描述：

```
--network_module=networks.dylora --network_dim=16 --network_args "conv_dim=16" "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "conv_dim=32" "conv_alpha=16" "unit=8"
```

例如，当使用dim=16、unit=4（如下所述）进行学习时，可以学习和提取4个rank的LoRA，即4、8、12和16。通过在每个提取的模型中生成图像并进行比较，可以选择最佳rank的LoRA。

其他选项与普通的LoRA相同。

*`unit`是本存储库的独有扩展，在DyLoRA中，由于预计相比同维度（rank）的普通LoRA，学习时间更长，因此将分割单位增加。

### 从DyLoRA模型中提取LoRA模型

请使用`networks`文件夹中的`extract_lora_from_dylora.py`。指定`unit`单位后，从DyLoRA模型中提取LoRA模型。

例如，命令行如下：

```powershell
python networks\extract_lora_from_dylora.py --model "foldername/dylora-model.safetensors" --save_to "foldername/dylora-model-split.safetensors" --unit 4
```

`--model` 参数用于指定DyLoRA模型文件。`--save_to` 参数用于指定要保存提取的模型的文件名（rank值将附加到文件名中）。`--unit` 参数用于指定DyLoRA训练时的`unit`。 

## 分层学习率

请参阅PR＃355了解详细信息。

您可以指定完整模型的25个块的权重。虽然第一个块没有对应的LoRA，但为了与分层LoRA应用等的兼容性，将其设为25个。此外，如果不扩展到conv2d3x3，则某些块中可能不存在LoRA，但为了统一描述，请始终指定25个值。

请在 `--network_args` 中指定以下参数。

- `down_lr_weight`：指定U-Net down blocks的学习率权重。可以指定以下内容：
  - 每个块的权重：指定12个数字，例如`"down_lr_weight=0,0,0,0,0,0,1,1,1,1,1,1"`
  - 从预设中指定：例如`"down_lr_weight=sine"`（使用正弦曲线指定权重）。可以指定sine、cosine、linear、reverse_linear、zeros。另外，添加 `+数字` 时，可以将指定的数字加上（变为0.25〜1.25）。
- `mid_lr_weight`：指定U-Net mid block的学习率权重。只需指定一个数字，例如 `"mid_lr_weight=0.5"`。
- `up_lr_weight`：指定U-Net up blocks的学习率权重。与down_lr_weight相同。
- 省略指定的部分将被视为1.0。另外，如果将权重设为0，则不会创建该块的LoRA模块。
- `block_lr_zero_threshold`：如果权重小于此值，则不会创建LoRA模块。默认值为0。

### 分层学习率命令行指定示例：


```powershell
--network_args "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5" "mid_lr_weight=2.0" "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5"

--network_args "block_lr_zero_threshold=0.1" "down_lr_weight=sine+.5" "mid_lr_weight=1.5" "up_lr_weight=cosine+.5"
```

###  Hierarchical Learning Rate指定的toml文件示例：

```toml
network_args = [ "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5", "mid_lr_weight=2.0", "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5",]

network_args = [ "block_lr_zero_threshold=0.1", "down_lr_weight=sine+.5", "mid_lr_weight=1.5", "up_lr_weight=cosine+.5", ]
```

## 层次结构维度（rank）

您可以指定完整模型的25个块的维度（rank）。与分层学习率一样，某些块可能不存在LoRA，但请始终指定25个值。

请在 `--network_args` 中指定以下参数：

- `block_dims`：指定每个块的维度（rank）。指定25个数字，例如 `"block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"`。
- `block_alphas`：指定每个块的alpha。与block_dims一样，指定25个数字。如果省略，将使用network_alpha的值。
- `conv_block_dims`：将LoRA扩展到Conv2d 3x3，并指定每个块的维度（rank）。
- `conv_block_alphas`：在将LoRA扩展到Conv2d 3x3时指定每个块的alpha。如果省略，将使用conv_alpha的值。

### 层次结构维度（rank）命令行指定示例：


```powershell
--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "conv_block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"
```

### 层级别dim(rank) toml文件指定示例：

```toml
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2",]
  
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2", "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",]
```

# Other scripts
这些是与LoRA相关的脚本，如合并脚本等。

关于合并脚本
您可以使用merge_lora.py脚本将LoRA的训练结果合并到稳定扩散模型中，也可以将多个LoRA模型合并。

合并到稳定扩散模型中的LoRA模型
合并后的模型可以像常规的稳定扩散ckpt一样使用。例如，以下是一个命令行示例：

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors --ratios 0.8
```

请使用 Stable Diffusion v2.x 模型进行训练并进行合并时，需要指定--v2选项。

使用--sd_model选项指定要合并的 Stable Diffusion 模型文件（仅支持 .ckpt 或 .safetensors 格式，目前不支持 Diffusers）。

使用--save_to选项指定合并后模型的保存路径（根据扩展名自动判断为 .ckpt 或 .safetensors）。

使用--models选项指定已训练的 LoRA 模型文件，也可以指定多个，然后按顺序进行合并。

使用--ratios选项以0~1.0的数字指定每个模型的应用率（将多大比例的权重反映到原始模型中）。例如，在接近过度拟合的情况下，降低应用率可能会使结果更好。请指定与模型数量相同的比率。 

当指定多个模型时，格式如下：


```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.8 0.5
```

### 将多个LoRA模型合并

将多个LoRA模型逐个应用于SD模型与将多个LoRA模型合并后再应用于SD模型之间，由于计算顺序的不同，会得到微妙不同的结果。

例如，下面是一个命令行示例：

```
python networks\merge_lora.py 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.6 0.4
```

--sd_model选项不需要指定。

通过--save_to选项指定合并后的LoRA模型的保存位置（.ckpt或.safetensors，根据扩展名自动识别）。

通过--models选项指定学习的LoRA模型文件。可以指定三个或更多。

通过--ratios选项以0~1.0的数字指定每个模型的比率（反映多少权重来自原始模型）。如果将两个模型一对一合并，则比率将是“0.5 0.5”。如果比率为“1.0 1.0”，则总重量将过大，可能会产生不理想的结果。

在v1和v2中学习的LoRA，以及rank（维数）或“alpha”不同的LoRA不能合并。仅包含U-Net的LoRA和包含U-Net+文本编码器的LoRA可以合并，但结果未知。

### 其他选项

* 精度
  * 可以从float、fp16或bf16中选择合并计算时的精度。默认为float以保证精度。如果想减少内存使用量，请指定fp16/bf16。
* save_precision
  * 可以从float、fp16或bf16中选择在保存模型时的精度。默认与精度相同。

## 合并多个维度不同的LoRA模型

将多个LoRA近似为一个LoRA（无法完全复制）。使用'svd_merge_lora.py'。例如，以下是命令行的示例。
```
python networks\svd_merge_lora.py 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors 
    --ratios 0.6 0.4 --new_rank 32 --device cuda
```
`merge_lora.py`和主要选项相同。以下选项已添加：

- `--new_rank`
  - 指定要创建的LoRA rank。
- `--new_conv_rank`
  - 指定要创建的Conv2d 3x3 LoRA的rank。如果省略，则与`new_rank`相同。
- `--device`
  - 如果指定为`--device cuda`，则在GPU上执行计算。处理速度将更快。

## 在此存储库中生成图像的脚本中

请在`gen_img_diffusers.py`中添加`--network_module`和`--network_weights`选项。其含义与训练时相同。

通过`--network_mul`选项，可以指定0~1.0的数字来改变LoRA的应用率。

## 请参考以下示例，在Diffusers的pipeline中生成。

所需文件仅为networks/lora.py。请注意，该示例只能在Diffusers版本0.10.2中正常运行。

```python
import torch
from diffusers import StableDiffusionPipeline
from networks.lora import LoRAModule, create_network_from_weights
from safetensors.torch import load_file

# if the ckpt is CompVis based, convert it to Diffusers beforehand with tools/convert_diffusers20_original_sd.py. See --help for more details.

model_id_or_dir = r"model_id_on_hugging_face_or_dir"
device = "cuda"

# create pipe
print(f"creating pipe from {model_id_or_dir}...")
pipe = StableDiffusionPipeline.from_pretrained(model_id_or_dir, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to(device)
vae = pipe.vae
text_encoder = pipe.text_encoder
unet = pipe.unet

# load lora networks
print(f"loading lora networks...")

lora_path1 = r"lora1.safetensors"
sd = load_file(lora_path1)   # If the file is .ckpt, use torch.load instead.
network1, sd = create_network_from_weights(0.5, None, vae, text_encoder,unet, sd)
network1.apply_to(text_encoder, unet)
network1.load_state_dict(sd)
network1.to(device, dtype=torch.float16)

# # You can merge weights instead of apply_to+load_state_dict. network.set_multiplier does not work
# network.merge_to(text_encoder, unet, sd)

lora_path2 = r"lora2.safetensors"
sd = load_file(lora_path2) 
network2, sd = create_network_from_weights(0.7, None, vae, text_encoder,unet, sd)
network2.apply_to(text_encoder, unet)
network2.load_state_dict(sd)
network2.to(device, dtype=torch.float16)

lora_path3 = r"lora3.safetensors"
sd = load_file(lora_path3)
network3, sd = create_network_from_weights(0.5, None, vae, text_encoder,unet, sd)
network3.apply_to(text_encoder, unet)
network3.load_state_dict(sd)
network3.to(device, dtype=torch.float16)

# prompts
prompt = "masterpiece, best quality, 1girl, in white shirt, looking at viewer"
negative_prompt = "bad quality, worst quality, bad anatomy, bad hands"

# exec pipe
print("generating image...")
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, negative_prompt=negative_prompt).images[0]

# if not merged, you can use set_multiplier
# network1.set_multiplier(0.8)
# and generate image again...

# save image
image.save(r"by_diffusers..png")
```

## 从两个模型的差异中创建LoRA模型。

[参考讨论链接](https://github.com/cloneofsimo/lora/discussions/56)這是參考實現的結果。數學公式沒有改變（我並不完全理解，但似乎使用奇異值分解進行了近似）。

将两个模型（例如微调原始模型和微调后的模型）的差异近似为LoRA。

### 脚本执行方法

请按以下方式指定。

```
python networks\extract_lora_from_models.py --model_org base-model.ckpt
    --model_tuned fine-tuned-model.ckpt 
    --save_to lora-weights.safetensors --dim 4
```

--model_org 选项指定原始的Stable Diffusion模型。如果要应用创建的LoRA模型，则需要指定该模型并将其应用。可以指定.ckpt或.safetensors文件。

--model_tuned 选项指定要提取差分的目标Stable Diffusion模型。例如，可以指定经过Fine Tuning或DreamBooth后的模型。可以指定.ckpt或.safetensors文件。

--save_to 指定LoRA模型的保存路径。--dim指定LoRA的维数。

生成的LoRA模型可以像已训练的LoRA模型一样使用。

当两个模型的文本编码器相同时，LoRA将成为仅包含U-Net的LoRA。

### 其他选项

- `--v2`
  - 如果使用v2.x的稳定扩散模型，请指定此选项。
- `--device`
  - 指定为 ``--device cuda`` 可在GPU上执行计算。这会使处理速度更快（即使在CPU上也不会太慢，大约快几倍）。
- `--save_precision`
  - 指定LoRA的保存格式为“float”、“fp16”、“bf16”。如果省略，将使用float。
- `--conv_dim`
  - 指定后，将扩展LoRA的应用范围到Conv2d 3x3。指定Conv2d 3x3的rank。
  - 
## 图像大小调整脚本

（稍后将整理文件，但现在先在这里写下说明。）

在 Aspect Ratio Bucketing 的功能扩展中，现在可以将小图像直接用作教师数据，而无需进行放大。我收到了一个用于前处理的脚本，其中包括将原始教师图像缩小的图像添加到教师数据中可以提高准确性的报告。我整理了这个脚本并加入了感谢 bmaltais 先生。

### 执行脚本的方法如下。
原始图像以及调整大小后的图像将保存到转换目标文件夹中。调整大小后的图像将在文件名中添加“+512x512”之类的调整后的分辨率（与图像大小不同）。小于调整大小后分辨率的图像将不会被放大。

```
python tools\resize_images_to_resolution.py --max_resolution 512x512,384x384,256x256 --save_as_png 
    --copy_associated_files 源图像文件夹目标文件夹
```

在元画像文件夹中的图像文件将被调整大小以达到指定的分辨率（可以指定多个），并保存到目标文件夹中。除图像外的文件将被保留为原样。

请使用“--max_resolution”选项指定调整大小后的大小，使其达到指定的面积大小。如果指定多个，则会在每个分辨率上进行调整大小。例如，“512x512，384x384，256x256”将使目标文件夹中的图像变为原始大小和调整大小后的大小×3共计4张图像。

如果使用“--save_as_png”选项，则会以PNG格式保存。如果省略，则默认以JPEG格式（quality=100）保存。

如果使用“--copy_associated_files”选项，则会将与图像相同的文件名（例如标题等）的文件复制到调整大小后的图像文件的文件名相同的位置，但不包括扩展名。

### 其他选项

- divisible_by
  - 将图像中心裁剪到能够被该值整除的大小（分别是垂直和水平的大小），以便调整大小后的图像大小可以被该值整除。
- interpolation
  - 指定缩小时的插值方法。可从``area、cubic、lanczos4``中选择，默认为``area``。


# 追加信息

## 与cloneofsimo的代码库的区别

截至2022年12月25日，本代码库将LoRA应用扩展到了Text Encoder的MLP、U-Net的FFN以及Transformer的输入/输出投影中，从而增强了表现力。但是，内存使用量增加了，接近了8GB的限制。

此外，模块交换机制也完全不同。

## 关于未来的扩展

除了LoRA之外，我们还计划添加其他扩展，以支持更多的功能。
