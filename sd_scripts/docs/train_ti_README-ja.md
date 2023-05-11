[Textual Inversion](https://textual-inversion.github.io/) の学習についての説明です。

[学習についての共通ドキュメント](./train_README-ja.md) もあわせてご覧ください。

実装に当たっては https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion を大いに参考にしました。

学習したモデルはWeb UIでもそのまま使えます。

# 学習の手順

あらかじめこのリポジトリのREADMEを参照し、環境整備を行ってください。

## データの準備

[学習データの準備について](./train_README-ja.md) を参照してください。

## 学習の実行

``train_textual_inversion.py`` を用います。以下はコマンドラインの例です（DreamBooth手法）。

```
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py 
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
    --token_string=mychar4 --init_word=cute --num_vectors_per_token=4
```

``--token_string`` に学習時のトークン文字列を指定します。__学習時のプロンプトは、この文字列を含むようにしてください（token_stringがmychar4なら、``mychar4 1girl`` など）__。プロンプトのこの文字列の部分が、Textual Inversionの新しいtokenに置換されて学習されます。DreamBooth, class+identifier形式のデータセットとして、`token_string` をトークン文字列にするのが最も簡単で確実です。

プロンプトにトークン文字列が含まれているかどうかは、``--debug_dataset`` で置換後のtoken idが表示されますので、以下のように ``49408`` 以降のtokenが存在するかどうかで確認できます。

```
input ids: tensor([[49406, 49408, 49409, 49410, 49411, 49412, 49413, 49414, 49415, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])
```

tokenizerがすでに持っている単語（一般的な単語）は使用できません。

``--init_word`` にembeddingsを初期化するときのコピー元トークンの文字列を指定します。学ばせたい概念が近いものを選ぶとよいようです。二つ以上のトークンになる文字列は指定できません。

``--num_vectors_per_token`` にいくつのトークンをこの学習で使うかを指定します。多いほうが表現力が増しますが、その分多くのトークンを消費します。たとえばnum_vectors_per_token=8の場合、指定したトークン文字列は（一般的なプロンプトの77トークン制限のうち）8トークンを消費します。

以上がTextual Inversionのための主なオプションです。以降は他の学習スクリプトと同様です。

`num_cpu_threads_per_process` には通常は1を指定するとよいようです。

`pretrained_model_name_or_path` に追加学習を行う元となるモデルを指定します。Stable Diffusionのcheckpointファイル（.ckptまたは.safetensors）、Diffusersのローカルディスクにあるモデルディレクトリ、DiffusersのモデルID（"stabilityai/stable-diffusion-2"など）が指定できます。

`output_dir` に学習後のモデルを保存するフォルダを指定します。`output_name` にモデルのファイル名を拡張子を除いて指定します。`save_model_as` でsafetensors形式での保存を指定しています。

`dataset_config` に `.toml` ファイルを指定します。ファイル内でのバッチサイズ指定は、当初はメモリ消費を抑えるために `1` としてください。

学習させるステップ数 `max_train_steps` を10000とします。学習率 `learning_rate` はここでは5e-6を指定しています。

省メモリ化のため `mixed_precision="fp16"` を指定します（RTX30 シリーズ以降では `bf16` も指定できます。環境整備時にaccelerateに行った設定と合わせてください）。また `gradient_checkpointing` を指定します。

オプティマイザ（モデルを学習データにあうように最適化＝学習させるクラス）にメモリ消費の少ない 8bit AdamW を使うため、 `optimizer_type="AdamW8bit"` を指定します。

`xformers` オプションを指定し、xformersのCrossAttentionを用います。xformersをインストールしていない場合やエラーとなる場合（環境にもよりますが `mixed_precision="no"` の場合など）、代わりに `mem_eff_attn` オプションを指定すると省メモリ版CrossAttentionを使用します（速度は遅くなります）。

ある程度メモリがある場合は、`.toml` ファイルを編集してバッチサイズをたとえば `8` くらいに増やしてください（高速化と精度向上の可能性があります）。

### よく使われるオプションについて

以下の場合にはオプションに関するドキュメントを参照してください。

- Stable Diffusion 2.xまたはそこからの派生モデルを学習する
- clip skipを2以上を前提としたモデルを学習する
- 75トークンを超えたキャプションで学習する

### Textual Inversionでのバッチサイズについて

モデル全体を学習するDreamBoothやfine tuningに比べてメモリ使用量が少ないため、バッチサイズは大きめにできます。

# Textual Inversionのその他の主なオプション

すべてのオプションについては別文書を参照してください。

* `--weights`
  * 学習前に学習済みのembeddingsを読み込み、そこから追加で学習します。
* `--use_object_template`
  * キャプションではなく既定の物体用テンプレート文字列（``a photo of a {}``など）で学習します。公式実装と同じになります。キャプションは無視されます。
* `--use_style_template`
  * キャプションではなく既定のスタイル用テンプレート文字列で学習します（``a painting in the style of {}``など）。公式実装と同じになります。キャプションは無視されます。

## 当リポジトリ内の画像生成スクリプトで生成する

gen_img_diffusers.pyに、``--textual_inversion_embeddings`` オプションで学習したembeddingsファイルを指定してください（複数可）。プロンプトでembeddingsファイルのファイル名（拡張子を除く）を使うと、そのembeddingsが適用されます。

