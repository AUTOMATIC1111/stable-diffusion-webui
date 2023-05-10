# LoRAの学習について

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（arxiv）、[LoRA](https://github.com/microsoft/LoRA)（github）をStable Diffusionに適用したものです。

[cloneofsimo氏のリポジトリ](https://github.com/cloneofsimo/lora)を大いに参考にさせていただきました。ありがとうございます。

通常のLoRAは Linear およぴカーネルサイズ 1x1 の Conv2d にのみ適用されますが、カーネルサイズ 3x3 のConv2dに適用を拡大することもできます。

Conv2d 3x3への拡大は [cloneofsimo氏](https://github.com/cloneofsimo/lora) が最初にリリースし、KohakuBlueleaf氏が [LoCon](https://github.com/KohakuBlueleaf/LoCon) でその有効性を明らかにしたものです。KohakuBlueleaf氏に深く感謝します。

8GB VRAMでもぎりぎり動作するようです。

[学習についての共通ドキュメント](./train_README-ja.md) もあわせてご覧ください。

# 学習できるLoRAの種類

以下の二種類をサポートします。以下は当リポジトリ内の独自の名称です。

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers、リエラと読みます)

    Linear およびカーネルサイズ 1x1 の Conv2d に適用されるLoRA

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers、セリアと読みます)

    1.に加え、カーネルサイズ 3x3 の Conv2d に適用されるLoRA

LoRA-LierLaに比べ、LoRA-C3Liarは適用される層が増える分、高い精度が期待できるかもしれません。

また学習時は __DyLoRA__ を使用することもできます（後述します）。

## 学習したモデルに関する注意

LoRA-LierLa は、AUTOMATIC1111氏のWeb UIのLoRA機能で使用することができます。

LoRA-C3Liarを使いWeb UIで生成するには、こちらの[WebUI用extension](https://github.com/kohya-ss/sd-webui-additional-networks)を使ってください。

いずれも学習したLoRAのモデルを、Stable Diffusionのモデルにこのリポジトリ内のスクリプトであらかじめマージすることもできます。

cloneofsimo氏のリポジトリ、およびd8ahazard氏の[Dreambooth Extension for Stable-Diffusion-WebUI](https://github.com/d8ahazard/sd_dreambooth_extension)とは、現時点では互換性がありません。いくつかの機能拡張を行っているためです（後述）。

# 学習の手順

あらかじめこのリポジトリのREADMEを参照し、環境整備を行ってください。

## データの準備

[学習データの準備について](./train_README-ja.md) を参照してください。


## 学習の実行

`train_network.py`を用います。

`train_network.py`では `--network_module` オプションに、学習対象のモジュール名を指定します。LoRAに対応するのは`network.lora`となりますので、それを指定してください。

なお学習率は通常のDreamBoothやfine tuningよりも高めの、`1e-4`～`1e-3`程度を指定するとよいようです。

以下はコマンドラインの例です。

```
accelerate launch --num_cpu_threads_per_process 1 train_network.py 
    --pretrained_model_name_or_path=<.ckptまたは.safetensordまたはDiffusers版モデルのディレクトリ> 
    --dataset_config=<データ準備で作成した.tomlファイル> 
    --output_dir=<学習したモデルの出力先フォルダ>  
    --output_name=<学習したモデル出力時のファイル名> 
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

このコマンドラインでは LoRA-LierLa が学習されます。

`--output_dir` オプションで指定したフォルダに、LoRAのモデルが保存されます。他のオプション、オプティマイザ等については [学習の共通ドキュメント](./train_README-ja.md) の「よく使われるオプション」も参照してください。

その他、以下のオプションが指定できます。

* `--network_dim`
  * LoRAのRANKを指定します（``--networkdim=4``など）。省略時は4になります。数が多いほど表現力は増しますが、学習に必要なメモリ、時間は増えます。また闇雲に増やしても良くないようです。
* `--network_alpha`
  *  アンダーフローを防ぎ安定して学習するための ``alpha`` 値を指定します。デフォルトは1です。``network_dim``と同じ値を指定すると以前のバージョンと同じ動作になります。
* `--persistent_data_loader_workers`
  * Windows環境で指定するとエポック間の待ち時間が大幅に短縮されます。
* `--max_data_loader_n_workers`
  * データ読み込みのプロセス数を指定します。プロセス数が多いとデータ読み込みが速くなりGPUを効率的に利用できますが、メインメモリを消費します。デフォルトは「`8` または `CPU同時実行スレッド数-1` の小さいほう」なので、メインメモリに余裕がない場合や、GPU使用率が90%程度以上なら、それらの数値を見ながら `2` または `1` 程度まで下げてください。
* `--network_weights`
  * 学習前に学習済みのLoRAの重みを読み込み、そこから追加で学習します。
* `--network_train_unet_only`
  * U-Netに関連するLoRAモジュールのみ有効とします。fine tuning的な学習で指定するとよいかもしれません。
* `--network_train_text_encoder_only`
  * Text Encoderに関連するLoRAモジュールのみ有効とします。Textual Inversion的な効果が期待できるかもしれません。
* `--unet_lr`
  * U-Netに関連するLoRAモジュールに、通常の学習率（--learning_rateオプションで指定）とは異なる学習率を使う時に指定します。
* `--text_encoder_lr`
  * Text Encoderに関連するLoRAモジュールに、通常の学習率（--learning_rateオプションで指定）とは異なる学習率を使う時に指定します。Text Encoderのほうを若干低めの学習率（5e-5など）にしたほうが良い、という話もあるようです。
* `--network_args`
  * 複数の引数を指定できます。後述します。

`--network_train_unet_only` と `--network_train_text_encoder_only` の両方とも未指定時（デフォルト）はText EncoderとU-Netの両方のLoRAモジュールを有効にします。

# その他の学習方法

## LoRA-C3Lier を学習する

`--network_args` に以下のように指定してください。`conv_dim` で Conv2d (3x3) の rank を、`conv_alpha` で alpha を指定してください。

```
--network_args "conv_dim=4" "conv_alpha=1"
```

以下のように alpha 省略時は1になります。

```
--network_args "conv_dim=4"
```

## DyLoRA

DyLoRAはこちらの論文で提案されたものです。[DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation](https://arxiv.org/abs/2210.07558)　公式実装は[こちら](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)です。

論文によると、LoRAのrankは必ずしも高いほうが良いわけではなく、対象のモデル、データセット、タスクなどにより適切なrankを探す必要があるようです。DyLoRAを使うと、指定したdim(rank)以下のさまざまなrankで同時にLoRAを学習します。これにより最適なrankをそれぞれ学習して探す手間を省くことができます。

当リポジトリの実装は公式実装をベースに独自の拡張を加えています（そのため不具合などあるかもしれません）。

### 当リポジトリのDyLoRAの特徴

学習後のDyLoRAのモデルファイルはLoRAと互換性があります。また、モデルファイルから指定したdim(rank)以下の複数のdimのLoRAを抽出できます。

DyLoRA-LierLa、DyLoRA-C3Lierのどちらも学習できます。

### DyLoRAで学習する

`--network_module=networks.dylora` のように、DyLoRAに対応する`network.dylora`を指定してください。

また `--network_args` に、たとえば`--network_args "unit=4"`のように`unit`を指定します。`unit`はrankを分割する単位です。たとえば`--network_dim=16 --network_args "unit=4"` のように指定します。`unit`は`network_dim`を割り切れる値（`network_dim`は`unit`の倍数）としてください。

`unit`を指定しない場合は、`unit=1`として扱われます。

記述例は以下です。

```
--network_module=networks.dylora --network_dim=16 --network_args "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "unit=4"
```

DyLoRA-C3Lierの場合は、`--network_args` に`"conv_dim=4"`のように`conv_dim`を指定します。通常のLoRAと異なり、`conv_dim`は`network_dim`と同じ値である必要があります。記述例は以下です。

```
--network_module=networks.dylora --network_dim=16 --network_args "conv_dim=16" "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "conv_dim=32" "conv_alpha=16" "unit=8"
```

たとえばdim=16、unit=4（後述）で学習すると、4、8、12、16の4つのrankのLoRAを学習、抽出できます。抽出した各モデルで画像を生成し、比較することで、最適なrankのLoRAを選択できます。

その他のオプションは通常のLoRAと同じです。

※ `unit`は当リポジトリの独自拡張で、DyLoRAでは同dim(rank)の通常LoRAに比べると学習時間が長くなることが予想されるため、分割単位を大きくしたものです。

### DyLoRAのモデルからLoRAモデルを抽出する

`networks`フォルダ内の `extract_lora_from_dylora.py`を使用します。指定した`unit`単位で、DyLoRAのモデルからLoRAのモデルを抽出します。

コマンドラインはたとえば以下のようになります。

```powershell
python networks\extract_lora_from_dylora.py --model "foldername/dylora-model.safetensors" --save_to "foldername/dylora-model-split.safetensors" --unit 4
```

`--model` にはDyLoRAのモデルファイルを指定します。`--save_to` には抽出したモデルを保存するファイル名を指定します（rankの数値がファイル名に付加されます）。`--unit` にはDyLoRAの学習時の`unit`を指定します。

## 階層別学習率

詳細は[PR #355](https://github.com/kohya-ss/sd-scripts/pull/355) をご覧ください。

フルモデルの25個のブロックの重みを指定できます。最初のブロックに該当するLoRAは存在しませんが、階層別LoRA適用等との互換性のために25個としています。またconv2d3x3に拡張しない場合も一部のブロックにはLoRAが存在しませんが、記述を統一するため常に25個の値を指定してください。

`--network_args` で以下の引数を指定してください。

- `down_lr_weight` : U-Netのdown blocksの学習率の重みを指定します。以下が指定可能です。
  - ブロックごとの重み : `"down_lr_weight=0,0,0,0,0,0,1,1,1,1,1,1"` のように12個の数値を指定します。
  - プリセットからの指定 : `"down_lr_weight=sine"` のように指定します（サインカーブで重みを指定します）。sine, cosine, linear, reverse_linear, zeros が指定可能です。また `"down_lr_weight=cosine+.25"` のように `+数値` を追加すると、指定した数値を加算します（0.25~1.25になります）。
- `mid_lr_weight` : U-Netのmid blockの学習率の重みを指定します。`"down_lr_weight=0.5"` のように数値を一つだけ指定します。
- `up_lr_weight` : U-Netのup blocksの学習率の重みを指定します。down_lr_weightと同様です。
- 指定を省略した部分は1.0として扱われます。また重みを0にするとそのブロックのLoRAモジュールは作成されません。
- `block_lr_zero_threshold` : 重みがこの値以下の場合、LoRAモジュールを作成しません。デフォルトは0です。

### 階層別学習率コマンドライン指定例:

```powershell
--network_args "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5" "mid_lr_weight=2.0" "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5"

--network_args "block_lr_zero_threshold=0.1" "down_lr_weight=sine+.5" "mid_lr_weight=1.5" "up_lr_weight=cosine+.5"
```

###  階層別学習率tomlファイル指定例:

```toml
network_args = [ "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5", "mid_lr_weight=2.0", "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5",]

network_args = [ "block_lr_zero_threshold=0.1", "down_lr_weight=sine+.5", "mid_lr_weight=1.5", "up_lr_weight=cosine+.5", ]
```

## 階層別dim (rank)

フルモデルの25個のブロックのdim (rank)を指定できます。階層別学習率と同様に一部のブロックにはLoRAが存在しない場合がありますが、常に25個の値を指定してください。

`--network_args` で以下の引数を指定してください。

- `block_dims` : 各ブロックのdim (rank)を指定します。`"block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"` のように25個の数値を指定します。
- `block_alphas` : 各ブロックのalphaを指定します。block_dimsと同様に25個の数値を指定します。省略時はnetwork_alphaの値が使用されます。
- `conv_block_dims` : LoRAをConv2d 3x3に拡張し、各ブロックのdim (rank)を指定します。
- `conv_block_alphas` : LoRAをConv2d 3x3に拡張したときの各ブロックのalphaを指定します。省略時はconv_alphaの値が使用されます。

###  階層別dim (rank)コマンドライン指定例:

```powershell
--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "conv_block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"
```

###  階層別dim (rank)tomlファイル指定例:

```toml
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2",]
  
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2", "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",]
```

# その他のスクリプト

マージ等LoRAに関連するスクリプト群です。

## マージスクリプトについて

merge_lora.pyでStable DiffusionのモデルにLoRAの学習結果をマージしたり、複数のLoRAモデルをマージしたりできます。

### Stable DiffusionのモデルにLoRAのモデルをマージする

マージ後のモデルは通常のStable Diffusionのckptと同様に扱えます。たとえば以下のようなコマンドラインになります。

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors --ratios 0.8
```

Stable Diffusion v2.xのモデルで学習し、それにマージする場合は、--v2オプションを指定してください。

--sd_modelオプションにマージの元となるStable Diffusionのモデルファイルを指定します（.ckptまたは.safetensorsのみ対応で、Diffusersは今のところ対応していません）。

--save_toオプションにマージ後のモデルの保存先を指定します（.ckptまたは.safetensors、拡張子で自動判定）。

--modelsに学習したLoRAのモデルファイルを指定します。複数指定も可能で、その時は順にマージします。

--ratiosにそれぞれのモデルの適用率（どのくらい重みを元モデルに反映するか）を0~1.0の数値で指定します。例えば過学習に近いような場合は、適用率を下げるとマシになるかもしれません。モデルの数と同じだけ指定してください。

複数指定時は以下のようになります。

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.8 0.5
```

### 複数のLoRAのモデルをマージする

複数のLoRAモデルをひとつずつSDモデルに適用する場合と、複数のLoRAモデルをマージしてからSDモデルにマージする場合とは、計算順序の関連で微妙に異なる結果になります。

たとえば以下のようなコマンドラインになります。

```
python networks\merge_lora.py 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.6 0.4
```

--sd_modelオプションは指定不要です。

--save_toオプションにマージ後のLoRAモデルの保存先を指定します（.ckptまたは.safetensors、拡張子で自動判定）。

--modelsに学習したLoRAのモデルファイルを指定します。三つ以上も指定可能です。

--ratiosにそれぞれのモデルの比率（どのくらい重みを元モデルに反映するか）を0~1.0の数値で指定します。二つのモデルを一対一でマージす場合は、「0.5 0.5」になります。「1.0 1.0」では合計の重みが大きくなりすぎて、恐らく結果はあまり望ましくないものになると思われます。

v1で学習したLoRAとv2で学習したLoRA、rank（次元数）や``alpha``の異なるLoRAはマージできません。U-NetだけのLoRAとU-Net+Text EncoderのLoRAはマージできるはずですが、結果は未知数です。


### その他のオプション

* precision
  * マージ計算時の精度をfloat、fp16、bf16から指定できます。省略時は精度を確保するためfloatになります。メモリ使用量を減らしたい場合はfp16/bf16を指定してください。
* save_precision
  * モデル保存時の精度をfloat、fp16、bf16から指定できます。省略時はprecisionと同じ精度になります。


## 複数のrankが異なるLoRAのモデルをマージする

複数のLoRAをひとつのLoRAで近似します（完全な再現はできません）。`svd_merge_lora.py`を用います。たとえば以下のようなコマンドラインになります。

```
python networks\svd_merge_lora.py 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors 
    --ratios 0.6 0.4 --new_rank 32 --device cuda
```

`merge_lora.py` と主なオプションは同一です。以下のオプションが追加されています。

- `--new_rank`
  - 作成するLoRAのrankを指定します。
- `--new_conv_rank`
  - 作成する Conv2d 3x3 LoRA の rank を指定します。省略時は `new_rank` と同じになります。
- `--device`
  - `--device cuda`としてcudaを指定すると計算をGPU上で行います。処理が速くなります。

## 当リポジトリ内の画像生成スクリプトで生成する

gen_img_diffusers.pyに、--network_module、--network_weightsの各オプションを追加してください。意味は学習時と同様です。

--network_mulオプションで0~1.0の数値を指定すると、LoRAの適用率を変えられます。

## Diffusersのpipelineで生成する

以下の例を参考にしてください。必要なファイルはnetworks/lora.pyのみです。Diffusersのバージョンは0.10.2以外では動作しない可能性があります。

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

## 二つのモデルの差分からLoRAモデルを作成する

[こちらのディスカッション](https://github.com/cloneofsimo/lora/discussions/56)を参考に実装したものです。数式はそのまま使わせていただきました（よく理解していませんが近似には特異値分解を用いるようです）。

二つのモデル（たとえばfine tuningの元モデルとfine tuning後のモデル）の差分を、LoRAで近似します。

### スクリプトの実行方法

以下のように指定してください。
```
python networks\extract_lora_from_models.py --model_org base-model.ckpt
    --model_tuned fine-tuned-model.ckpt 
    --save_to lora-weights.safetensors --dim 4
```

--model_orgオプションに元のStable Diffusionモデルを指定します。作成したLoRAモデルを適用する場合は、このモデルを指定して適用することになります。.ckptまたは.safetensorsが指定できます。

--model_tunedオプションに差分を抽出する対象のStable Diffusionモデルを指定します。たとえばfine tuningやDreamBooth後のモデルを指定します。.ckptまたは.safetensorsが指定できます。

--save_toにLoRAモデルの保存先を指定します。--dimにLoRAの次元数を指定します。

生成されたLoRAモデルは、学習したLoRAモデルと同様に使用できます。

Text Encoderが二つのモデルで同じ場合にはLoRAはU-NetのみのLoRAとなります。

### その他のオプション

- `--v2`
  - v2.xのStable Diffusionモデルを使う場合に指定してください。
- `--device`
  - ``--device cuda``としてcudaを指定すると計算をGPU上で行います。処理が速くなります（CPUでもそこまで遅くないため、せいぜい倍～数倍程度のようです）。
- `--save_precision`
  - LoRAの保存形式を"float", "fp16", "bf16"から指定します。省略時はfloatになります。
- `--conv_dim`
  - 指定するとLoRAの適用範囲を Conv2d 3x3 へ拡大します。Conv2d 3x3 の rank を指定します。

## 画像リサイズスクリプト

（のちほどドキュメントを整理しますがとりあえずここに説明を書いておきます。）

Aspect Ratio Bucketingの機能拡張で、小さな画像については拡大しないでそのまま教師データとすることが可能になりました。元の教師画像を縮小した画像を、教師データに加えると精度が向上したという報告とともに前処理用のスクリプトをいただきましたので整備して追加しました。bmaltais氏に感謝します。

### スクリプトの実行方法

以下のように指定してください。元の画像そのまま、およびリサイズ後の画像が変換先フォルダに保存されます。リサイズ後の画像には、ファイル名に ``+512x512`` のようにリサイズ先の解像度が付け加えられます（画像サイズとは異なります）。リサイズ先の解像度より小さい画像は拡大されることはありません。

```
python tools\resize_images_to_resolution.py --max_resolution 512x512,384x384,256x256 --save_as_png 
    --copy_associated_files 元画像フォルダ 変換先フォルダ
```

元画像フォルダ内の画像ファイルが、指定した解像度（複数指定可）と同じ面積になるようにリサイズされ、変換先フォルダに保存されます。画像以外のファイルはそのままコピーされます。

``--max_resolution`` オプションにリサイズ先のサイズを例のように指定してください。面積がそのサイズになるようにリサイズします。複数指定すると、それぞれの解像度でリサイズされます。``512x512,384x384,256x256``なら、変換先フォルダの画像は、元サイズとリサイズ後サイズ×3の計4枚になります。

``--save_as_png`` オプションを指定するとpng形式で保存します。省略するとjpeg形式（quality=100）で保存されます。

``--copy_associated_files`` オプションを指定すると、拡張子を除き画像と同じファイル名（たとえばキャプションなど）のファイルが、リサイズ後の画像のファイル名と同じ名前でコピーされます。


### その他のオプション

- divisible_by
  - リサイズ後の画像のサイズ（縦、横のそれぞれ）がこの値で割り切れるように、画像中心を切り出します。
- interpolation
  - 縮小時の補完方法を指定します。``area, cubic, lanczos4``から選択可能で、デフォルトは``area``です。


# 追加情報

## cloneofsimo氏のリポジトリとの違い

2022/12/25時点では、当リポジトリはLoRAの適用個所をText EncoderのMLP、U-NetのFFN、Transformerのin/out projectionに拡大し、表現力が増しています。ただその代わりメモリ使用量は増え、8GBぎりぎりになりました。

またモジュール入れ替え機構は全く異なります。

## 将来拡張について

LoRAだけでなく他の拡張にも対応可能ですので、それらも追加予定です。
