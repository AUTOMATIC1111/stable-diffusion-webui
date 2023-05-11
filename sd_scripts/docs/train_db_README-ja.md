DreamBoothのガイドです。

[学習についての共通ドキュメント](./train_README-ja.md) もあわせてご覧ください。

# 概要

DreamBoothとは、画像生成モデルに特定の主題を追加学習し、それを特定の識別子で生成する技術です。[論文はこちら](https://arxiv.org/abs/2208.12242)。

具体的には、Stable Diffusionのモデルにキャラや画風などを学ばせ、それを `shs` のような特定の単語で呼び出せる（生成画像に出現させる）ことができます。

スクリプトは[DiffusersのDreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)を元にしていますが、以下のような機能追加を行っています（いくつかの機能は元のスクリプト側もその後対応しています）。

スクリプトの主な機能は以下の通りです。

- 8bit Adam optimizerおよびlatentのキャッシュによる省メモリ化（[Shivam Shrirao氏版](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth)と同様）。
- xformersによる省メモリ化。
- 512x512だけではなく任意サイズでの学習。
- augmentationによる品質の向上。
- DreamBoothだけではなくText Encoder+U-Netのfine tuningに対応。
- Stable Diffusion形式でのモデルの読み書き。
- Aspect Ratio Bucketing。
- Stable Diffusion v2.0対応。

# 学習の手順

あらかじめこのリポジトリのREADMEを参照し、環境整備を行ってください。

## データの準備

[学習データの準備について](./train_README-ja.md) を参照してください。

## 学習の実行

スクリプトを実行します。最大限、メモリを節約したコマンドは以下のようになります（実際には1行で入力します）。それぞれの行を必要に応じて書き換えてください。12GB程度のVRAMで動作するようです。

```
accelerate launch --num_cpu_threads_per_process 1 train_db.py 
    --pretrained_model_name_or_path=<.ckptまたは.safetensordまたはDiffusers版モデルのディレクトリ> 
    --dataset_config=<データ準備で作成した.tomlファイル> 
    --output_dir=<学習したモデルの出力先フォルダ>  
    --output_name=<学習したモデル出力時のファイル名> 
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

`num_cpu_threads_per_process` には通常は1を指定するとよいようです。

`pretrained_model_name_or_path` に追加学習を行う元となるモデルを指定します。Stable Diffusionのcheckpointファイル（.ckptまたは.safetensors）、Diffusersのローカルディスクにあるモデルディレクトリ、DiffusersのモデルID（"stabilityai/stable-diffusion-2"など）が指定できます。

`output_dir` に学習後のモデルを保存するフォルダを指定します。`output_name` にモデルのファイル名を拡張子を除いて指定します。`save_model_as` でsafetensors形式での保存を指定しています。

`dataset_config` に `.toml` ファイルを指定します。ファイル内でのバッチサイズ指定は、当初はメモリ消費を抑えるために `1` としてください。

`prior_loss_weight` は正則化画像のlossの重みです。通常は1.0を指定します。

学習させるステップ数 `max_train_steps` を1600とします。学習率 `learning_rate` はここでは1e-6を指定しています。

省メモリ化のため `mixed_precision="fp16"` を指定します（RTX30 シリーズ以降では `bf16` も指定できます。環境整備時にaccelerateに行った設定と合わせてください）。また `gradient_checkpointing` を指定します。

オプティマイザ（モデルを学習データにあうように最適化＝学習させるクラス）にメモリ消費の少ない 8bit AdamW を使うため、 `optimizer_type="AdamW8bit"` を指定します。

`xformers` オプションを指定し、xformersのCrossAttentionを用います。xformersをインストールしていない場合やエラーとなる場合（環境にもよりますが `mixed_precision="no"` の場合など）、代わりに `mem_eff_attn` オプションを指定すると省メモリ版CrossAttentionを使用します（速度は遅くなります）。

省メモリ化のため `cache_latents` オプションを指定してVAEの出力をキャッシュします。

ある程度メモリがある場合は、`.toml` ファイルを編集してバッチサイズをたとえば `4` くらいに増やしてください（高速化と精度向上の可能性があります）。また `cache_latents` を外すことで augmentation が可能になります。

### よく使われるオプションについて

以下の場合には [学習の共通ドキュメント](./train_README-ja.md) の「よく使われるオプション」を参照してください。

- Stable Diffusion 2.xまたはそこからの派生モデルを学習する
- clip skipを2以上を前提としたモデルを学習する
- 75トークンを超えたキャプションで学習する

### DreamBoothでのステップ数について

当スクリプトでは省メモリ化のため、ステップ当たりの学習回数が元のスクリプトの半分になっています（対象の画像と正則化画像を同一のバッチではなく別のバッチに分割して学習するため）。

元のDiffusers版やXavierXiao氏のStable Diffusion版とほぼ同じ学習を行うには、ステップ数を倍にしてください。

（学習画像と正則化画像をまとめてから shuffle するため厳密にはデータの順番が変わってしまいますが、学習には大きな影響はないと思います。）

### DreamBoothでのバッチサイズについて

モデル全体を学習するためLoRA等の学習に比べるとメモリ消費量は多くなります（fine tuningと同じ）。

### 学習率について

Diffusers版では5e-6ですがStable Diffusion版は1e-6ですので、上のサンプルでは1e-6を指定しています。

### 以前の形式のデータセット指定をした場合のコマンドライン

解像度やバッチサイズをオプションで指定します。コマンドラインの例は以下の通りです。

```
accelerate launch --num_cpu_threads_per_process 1 train_db.py 
    --pretrained_model_name_or_path=<.ckptまたは.safetensordまたはDiffusers版モデルのディレクトリ> 
    --train_data_dir=<学習用データのディレクトリ> 
    --reg_data_dir=<正則化画像のディレクトリ> 
    --output_dir=<学習したモデルの出力先ディレクトリ> 
    --output_name=<学習したモデル出力時のファイル名> 
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

## 学習したモデルで画像生成する

学習が終わると指定したフォルダに指定した名前でsafetensorsファイルが出力されます。

v1.4/1.5およびその他の派生モデルの場合、このモデルでAutomatic1111氏のWebUIなどで推論できます。models\Stable-diffusionフォルダに置いてください。

v2.xモデルでWebUIで画像生成する場合、モデルの仕様が記述された.yamlファイルが別途必要になります。v2.x baseの場合はv2-inference.yamlを、768/vの場合はv2-inference-v.yamlを、同じフォルダに置き、拡張子の前の部分をモデルと同じ名前にしてください。

![image](https://user-images.githubusercontent.com/52813779/210776915-061d79c3-6582-42c2-8884-8b91d2f07313.png)

各yamlファイルは[Stability AIのSD2.0のリポジトリ](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion)にあります。

# DreamBooth特有のその他の主なオプション

すべてのオプションについては別文書を参照してください。

## Text Encoderの学習を途中から行わない --stop_text_encoder_training

stop_text_encoder_trainingオプションに数値を指定すると、そのステップ数以降はText Encoderの学習を行わずU-Netだけ学習します。場合によっては精度の向上が期待できるかもしれません。

（恐らくText Encoderだけ先に過学習することがあり、それを防げるのではないかと推測していますが、詳細な影響は不明です。）

## Tokenizerのパディングをしない --no_token_padding
no_token_paddingオプションを指定するとTokenizerの出力をpaddingしません（Diffusers版の旧DreamBoothと同じ動きになります）。


<!-- 
bucketing（後述）を利用しかつaugmentation（後述）を使う場合の例は以下のようになります。

```
accelerate launch --num_cpu_threads_per_process 8 train_db.py 
    --pretrained_model_name_or_path=<.ckptまたは.safetensordまたはDiffusers版モデルのディレクトリ> 
    --train_data_dir=<学習用データのディレクトリ> 
    --reg_data_dir=<正則化画像のディレクトリ> 
    --output_dir=<学習したモデルの出力先ディレクトリ> 
    --resolution=768,512 
    --train_batch_size=20 --learning_rate=5e-6 --max_train_steps=800 
    --use_8bit_adam --xformers --mixed_precision="bf16" 
    --save_every_n_epochs=1 --save_state --save_precision="bf16" 
    --logging_dir=logs 
    --enable_bucket --min_bucket_reso=384 --max_bucket_reso=1280 
    --color_aug --flip_aug --gradient_checkpointing --seed 42
```


-->
