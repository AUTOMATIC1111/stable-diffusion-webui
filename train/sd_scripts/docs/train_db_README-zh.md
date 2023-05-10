这是DreamBooth的指南。

请同时查看[关于学习的通用文档](./train_README-zh.md)。

# 概要

DreamBooth是一种将特定主题添加到图像生成模型中进行学习，并使用特定识别子生成它的技术。论文链接。

具体来说，它可以将角色和绘画风格等添加到Stable Diffusion模型中进行学习，并使用特定的单词（例如`shs`）来调用（呈现在生成的图像中）。

脚本基于Diffusers的DreamBooth，但添加了以下功能（一些功能已在原始脚本中得到支持）。

脚本的主要功能如下：

- 使用8位Adam优化器和潜在变量的缓存来节省内存（与Shivam Shrirao版相似）。
- 使用xformers来节省内存。
- 不仅支持512x512，还支持任意尺寸的训练。
- 通过数据增强来提高质量。
- 支持DreamBooth和Text Encoder + U-Net的微调。
- 支持以Stable Diffusion格式读写模型。
- 支持Aspect Ratio Bucketing。
- 支持Stable Diffusion v2.0。

# 训练步骤

请先参阅此存储库的README以进行环境设置。

## 准备数据

请参阅[有关准备训练数据的说明](./train_README-zh.md)。

## 运行训练

运行脚本。以下是最大程度地节省内存的命令（实际上，这将在一行中输入）。请根据需要修改每行。它似乎需要约12GB的VRAM才能运行。
```
accelerate launch --num_cpu_threads_per_process 1 train_db.py 
    --pretrained_model_name_or_path=<.ckpt或.safetensord或Diffusers版模型的目录>
    --dataset_config=<数据准备时创建的.toml文件>
    --output_dir=<训练模型的输出目录>
    --output_name=<训练模型输出时的文件名>
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=1600 
    --learning_rate=1e-6 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
```
`num_cpu_threads_per_process` 通常应该设置为1。

`pretrained_model_name_or_path` 指定要进行追加训练的基础模型。可以指定 Stable Diffusion 的 checkpoint 文件（.ckpt 或 .safetensors）、Diffusers 的本地模型目录或模型 ID（如 "stabilityai/stable-diffusion-2"）。

`output_dir` 指定保存训练后模型的文件夹。在 `output_name` 中指定模型文件名，不包括扩展名。使用 `save_model_as` 指定以 safetensors 格式保存。

在 `dataset_config` 中指定 `.toml` 文件。初始批处理大小应为 `1`，以减少内存消耗。

`prior_loss_weight` 是正则化图像损失的权重。通常设为1.0。

将要训练的步数 `max_train_steps` 设置为1600。在这里，学习率 `learning_rate` 被设置为1e-6。

为了节省内存，设置 `mixed_precision="fp16"`（在 RTX30 系列及更高版本中也可以设置为 `bf16`）。同时指定 `gradient_checkpointing`。

为了使用内存消耗较少的 8bit AdamW 优化器（将模型优化为适合于训练数据的状态），指定 `optimizer_type="AdamW8bit"`。

指定 `xformers` 选项，并使用 xformers 的 CrossAttention。如果未安装 xformers 或出现错误（具体情况取决于环境，例如使用 `mixed_precision="no"`），则可以指定 `mem_eff_attn` 选项以使用省内存版的 CrossAttention（速度会变慢）。

为了节省内存，指定 `cache_latents` 选项以缓存 VAE 的输出。

如果有足够的内存，请编辑 `.toml` 文件将批处理大小增加到大约 `4`（可能会提高速度和精度）。此外，取消 `cache_latents` 选项可以进行数据增强。

### 常用选项

对于以下情况，请参阅“常用选项”部分。

- 学习 Stable Diffusion 2.x 或其衍生模型。
- 学习基于 clip skip 大于等于2的模型。
- 学习超过75个令牌的标题。

### 关于DreamBooth中的步数

为了实现省内存化，该脚本中每个步骤的学习次数减半（因为学习和正则化的图像在训练时被分为不同的批次）。

要进行与原始Diffusers版或XavierXiao的Stable Diffusion版几乎相同的学习，请将步骤数加倍。

（虽然在将学习图像和正则化图像整合后再打乱顺序，但我认为对学习没有太大影响。）

关于DreamBooth的批量大小

与像LoRA这样的学习相比，为了训练整个模型，内存消耗量会更大（与微调相同）。

关于学习率

在Diffusers版中，学习率为5e-6，而在Stable Diffusion版中为1e-6，因此在上面的示例中指定了1e-6。

当使用旧格式的数据集指定命令行时

使用选项指定分辨率和批量大小。命令行示例如下。
```
accelerate launch --num_cpu_threads_per_process 1 train_db.py 
    --pretrained_model_name_or_path=<.ckpt或.safetensord或Diffusers版模型的目录> 
    --train_data_dir=<训练数据的目录> 
    --reg_data_dir=<正则化图像的目录> 
    --output_dir=<训练后模型的输出目录> 
    --output_name=<训练后模型输出文件的名称>  
    --prior_loss_weight=1.0 
    --resolution=512 
    --train_batch_size=1 
    --learning_rate=1e-6 
    --max_train_steps=1600 
    --use_8bit_adam 
    --xformers 
    --mixed_precision="bf16" 
    --cache_latents
    --gradient_checkpointing
```

## 使用训练好的模型生成图像

训练完成后，将在指定的文件夹中以指定的名称输出safetensors文件。

对于v1.4/1.5和其他派生模型，可以在此模型中使用Automatic1111先生的WebUI进行推断。请将其放置在models\Stable-diffusion文件夹中。

对于使用v2.x模型在WebUI中生成图像的情况，需要单独的.yaml文件来描述模型的规格。对于v2.x base，需要v2-inference.yaml，对于768/v，则需要v2-inference-v.yaml。请将它们放置在相同的文件夹中，并将文件扩展名之前的部分命名为与模型相同的名称。
![image](https://user-images.githubusercontent.com/52813779/210776915-061d79c3-6582-42c2-8884-8b91d2f07313.png)

每个yaml文件都在[Stability AI的SD2.0存储库](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion)……之中。

# DreamBooth的其他主要选项

有关所有选项的详细信息，请参阅另一份文档。

## 不在中途开始对文本编码器进行训练 --stop_text_encoder_training

如果在stop_text_encoder_training选项中指定一个数字，则在该步骤之后，将不再对文本编码器进行训练，只会对U-Net进行训练。在某些情况下，可能会期望提高精度。

（我们推测可能会有时候仅仅文本编码器会过度学习，而这样做可以避免这种情况，但详细影响尚不清楚。）

## 不进行分词器的填充 --no_token_padding

如果指定no_token_padding选项，则不会对分词器的输出进行填充（与Diffusers版本的旧DreamBooth相同）。

<!-- 
如果使用分桶（bucketing）和数据增强（augmentation），则使用示例如下：
```
accelerate launch --num_cpu_threads_per_process 8 train_db.py 
    --pretrained_model_name_or_path=<.ckpt或.safetensord或Diffusers版模型的目录> 
    --train_data_dir=<训练数据的目录> 
    --reg_data_dir=<正则化图像的目录> 
    --output_dir=<训练后模型的输出目录>
    --resolution=768,512 
    --train_batch_size=20 --learning_rate=5e-6 --max_train_steps=800 
    --use_8bit_adam --xformers --mixed_precision="bf16" 
    --save_every_n_epochs=1 --save_state --save_precision="bf16" 
    --logging_dir=logs 
    --enable_bucket --min_bucket_reso=384 --max_bucket_reso=1280 
    --color_aug --flip_aug --gradient_checkpointing --seed 42
```


-->
