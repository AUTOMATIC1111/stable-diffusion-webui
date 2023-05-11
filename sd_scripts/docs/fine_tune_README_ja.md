NovelAIの提案した学習手法、自動キャプションニング、タグ付け、Windows＋VRAM 12GB（SD v1.xの場合）環境等に対応したfine tuningです。ここでfine tuningとは、モデルを画像とキャプションで学習することを指します（LoRAやTextual Inversion、Hypernetworksは含みません）

[学習についての共通ドキュメント](./train_README-ja.md) もあわせてご覧ください。

# 概要

Diffusersを用いてStable DiffusionのU-Netのfine tuningを行います。NovelAIの記事にある以下の改善に対応しています（Aspect Ratio BucketingについてはNovelAIのコードを参考にしましたが、最終的なコードはすべてオリジナルです）。

* CLIP（Text Encoder）の最後の層ではなく最後から二番目の層の出力を用いる。
* 正方形以外の解像度での学習（Aspect Ratio Bucketing） 。
* トークン長を75から225に拡張する。
* BLIPによるキャプショニング（キャプションの自動作成）、DeepDanbooruまたはWD14Taggerによる自動タグ付けを行う。
* Hypernetworkの学習にも対応する。
* Stable Diffusion v2.0（baseおよび768/v）に対応。
* VAEの出力をあらかじめ取得しディスクに保存しておくことで、学習の省メモリ化、高速化を図る。

デフォルトではText Encoderの学習は行いません。モデル全体のfine tuningではU-Netだけを学習するのが一般的なようです（NovelAIもそのようです）。オプション指定でText Encoderも学習対象とできます。

# 追加機能について

## CLIPの出力の変更

プロンプトを画像に反映するため、テキストの特徴量への変換を行うのがCLIP（Text Encoder）です。Stable DiffusionではCLIPの最後の層の出力を用いていますが、それを最後から二番目の層の出力を用いるよう変更できます。NovelAIによると、これによりより正確にプロンプトが反映されるようになるとのことです。
元のまま、最後の層の出力を用いることも可能です。

※Stable Diffusion 2.0では最後から二番目の層をデフォルトで使います。clip_skipオプションを指定しないでください。

## 正方形以外の解像度での学習

Stable Diffusionは512\*512で学習されていますが、それに加えて256\*1024や384\*640といった解像度でも学習します。これによりトリミングされる部分が減り、より正しくプロンプトと画像の関係が学習されることが期待されます。
学習解像度はパラメータとして与えられた解像度の面積（＝メモリ使用量）を超えない範囲で、64ピクセル単位で縦横に調整、作成されます。

機械学習では入力サイズをすべて統一するのが一般的ですが、特に制約があるわけではなく、実際は同一のバッチ内で統一されていれば大丈夫です。NovelAIの言うbucketingは、あらかじめ教師データを、アスペクト比に応じた学習解像度ごとに分類しておくことを指しているようです。そしてバッチを各bucket内の画像で作成することで、バッチの画像サイズを統一します。

## トークン長の75から225への拡張

Stable Diffusionでは最大75トークン（開始・終了を含むと77トークン）ですが、それを225トークンまで拡張します。
ただしCLIPが受け付ける最大長は75トークンですので、225トークンの場合、単純に三分割してCLIPを呼び出してから結果を連結しています。

※これが望ましい実装なのかどうかはいまひとつわかりません。とりあえず動いてはいるようです。特に2.0では何も参考になる実装がないので独自に実装してあります。

※Automatic1111氏のWeb UIではカンマを意識して分割、といったこともしているようですが、私の場合はそこまでしておらず単純な分割です。

# 学習の手順

あらかじめこのリポジトリのREADMEを参照し、環境整備を行ってください。

## データの準備

[学習データの準備について](./train_README-ja.md) を参照してください。fine tuningではメタデータを用いるfine tuning方式のみ対応しています。

## 学習の実行
たとえば以下のように実行します。以下は省メモリ化のための設定です。それぞれの行を必要に応じて書き換えてください。

```
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py 
    --pretrained_model_name_or_path=<.ckptまたは.safetensordまたはDiffusers版モデルのディレクトリ> 
    --output_dir=<学習したモデルの出力先フォルダ>  
    --output_name=<学習したモデル出力時のファイル名> 
    --dataset_config=<データ準備で作成した.tomlファイル> 
    --save_model_as=safetensors 
    --learning_rate=5e-6 --max_train_steps=10000 
    --use_8bit_adam --xformers --gradient_checkpointing
    --mixed_precision=fp16
```

`num_cpu_threads_per_process` には通常は1を指定するとよいようです。

`pretrained_model_name_or_path` に追加学習を行う元となるモデルを指定します。Stable Diffusionのcheckpointファイル（.ckptまたは.safetensors）、Diffusersのローカルディスクにあるモデルディレクトリ、DiffusersのモデルID（"stabilityai/stable-diffusion-2"など）が指定できます。

`output_dir` に学習後のモデルを保存するフォルダを指定します。`output_name` にモデルのファイル名を拡張子を除いて指定します。`save_model_as` でsafetensors形式での保存を指定しています。

`dataset_config` に `.toml` ファイルを指定します。ファイル内でのバッチサイズ指定は、当初はメモリ消費を抑えるために `1` としてください。

学習させるステップ数 `max_train_steps` を10000とします。学習率 `learning_rate` はここでは5e-6を指定しています。

省メモリ化のため `mixed_precision="fp16"` を指定します（RTX30 シリーズ以降では `bf16` も指定できます。環境整備時にaccelerateに行った設定と合わせてください）。また `gradient_checkpointing` を指定します。

オプティマイザ（モデルを学習データにあうように最適化＝学習させるクラス）にメモリ消費の少ない 8bit AdamW を使うため、 `optimizer_type="AdamW8bit"` を指定します。

`xformers` オプションを指定し、xformersのCrossAttentionを用います。xformersをインストールしていない場合やエラーとなる場合（環境にもよりますが `mixed_precision="no"` の場合など）、代わりに `mem_eff_attn` オプションを指定すると省メモリ版CrossAttentionを使用します（速度は遅くなります）。

ある程度メモリがある場合は、`.toml` ファイルを編集してバッチサイズをたとえば `4` くらいに増やしてください（高速化と精度向上の可能性があります）。

### よく使われるオプションについて

以下の場合にはオプションに関するドキュメントを参照してください。

- Stable Diffusion 2.xまたはそこからの派生モデルを学習する
- clip skipを2以上を前提としたモデルを学習する
- 75トークンを超えたキャプションで学習する

### バッチサイズについて

モデル全体を学習するためLoRA等の学習に比べるとメモリ消費量は多くなります（DreamBoothと同じ）。

### 学習率について

1e-6から5e-6程度が一般的なようです。他のfine tuningの例なども参照してみてください。

### 以前の形式のデータセット指定をした場合のコマンドライン

解像度やバッチサイズをオプションで指定します。コマンドラインの例は以下の通りです。

```
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py 
    --pretrained_model_name_or_path=model.ckpt 
    --in_json meta_lat.json 
    --train_data_dir=train_data 
    --output_dir=fine_tuned 
    --shuffle_caption 
    --train_batch_size=1 --learning_rate=5e-6 --max_train_steps=10000 
    --use_8bit_adam --xformers --gradient_checkpointing
    --mixed_precision=bf16
    --save_every_n_epochs=4
```

<!-- 
### 勾配をfp16とした学習（実験的機能）
full_fp16オプションを指定すると勾配を通常のfloat32からfloat16（fp16）に変更して学習します（mixed precisionではなく完全なfp16学習になるようです）。これによりSD1.xの512*512サイズでは8GB未満、SD2.xの512*512サイズで12GB未満のVRAM使用量で学習できるようです。

あらかじめaccelerate configでfp16を指定し、オプションでmixed_precision="fp16"としてください（bf16では動作しません）。

メモリ使用量を最小化するためには、xformers、use_8bit_adam、gradient_checkpointingの各オプションを指定し、train_batch_sizeを1としてください。
（余裕があるようならtrain_batch_sizeを段階的に増やすと若干精度が上がるはずです。）

PyTorchのソースにパッチを当てて無理やり実現しています（PyTorch 1.12.1と1.13.0で確認）。精度はかなり落ちますし、途中で学習失敗する確率も高くなります。学習率やステップ数の設定もシビアなようです。それらを認識したうえで自己責任でお使いください。
-->

# fine tuning特有のその他の主なオプション

すべてのオプションについては別文書を参照してください。

## `train_text_encoder`
Text Encoderも学習対象とします。メモリ使用量が若干増加します。

通常のfine tuningではText Encoderは学習対象としませんが（恐らくText Encoderの出力に従うようにU-Netを学習するため）、学習データ数が少ない場合には、DreamBoothのようにText Encoder側に学習させるのも有効的なようです。

## `diffusers_xformers`
スクリプト独自のxformers置換機能ではなくDiffusersのxformers機能を利用します。Hypernetworkの学習はできなくなります。
