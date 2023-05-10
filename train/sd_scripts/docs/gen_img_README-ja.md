SD 1.xおよび2.xのモデル、当リポジトリで学習したLoRA、ControlNet（v1.0のみ動作確認）などに対応した、Diffusersベースの推論（画像生成）スクリプトです。コマンドラインから用います。

# 概要

* Diffusers (v0.10.2) ベースの推論（画像生成）スクリプト。
* SD 1.xおよび2.x (base/v-parameterization)モデルに対応。
* txt2img、img2img、inpaintingに対応。
* 対話モード、およびファイルからのプロンプト読み込み、連続生成に対応。
* プロンプト1行あたりの生成枚数を指定可能。
* 全体の繰り返し回数を指定可能。
* `fp16`だけでなく`bf16`にも対応。
* xformersに対応し高速生成が可能。
    * xformersにより省メモリ生成を行いますが、Automatic 1111氏のWeb UIほど最適化していないため、512*512の画像生成でおおむね6GB程度のVRAMを使用します。
* プロンプトの225トークンへの拡張。ネガティブプロンプト、重みづけに対応。
* Diffusersの各種samplerに対応（Web UIよりもsampler数は少ないです）。
* Text Encoderのclip skip（最後からn番目の層の出力を用いる）に対応。
* VAEの別途読み込み。
* CLIP Guided Stable Diffusion、VGG16 Guided Stable Diffusion、Highres. fix、upscale対応。
    * Highres. fixはWeb UIの実装を全く確認していない独自実装のため、出力結果は異なるかもしれません。
* LoRA対応。適用率指定、複数LoRA同時利用、重みのマージに対応。
    * Text EncoderとU-Netで別の適用率を指定することはできません。
* Attention Coupleに対応。
* ControlNet v1.0に対応。
* 途中でモデルを切り替えることはできませんが、バッチファイルを組むことで対応できます。
* 個人的に欲しくなった機能をいろいろ追加。

機能追加時にすべてのテストを行っているわけではないため、以前の機能に影響が出て一部機能が動かない可能性があります。何か問題があればお知らせください。

# 基本的な使い方

## 対話モードでの画像生成

以下のように入力してください。

```batchfile
python gen_img_diffusers.py --ckpt <モデル名> --outdir <画像出力先> --xformers --fp16 --interactive
```

`--ckpt`オプションにモデル（Stable Diffusionのcheckpointファイル、またはDiffusersのモデルフォルダ）、`--outdir`オプションに画像の出力先フォルダを指定します。

`--xformers`オプションでxformersの使用を指定します（xformersを使わない場合は外してください）。`--fp16`オプションでfp16（単精度）での推論を行います。RTX 30系のGPUでは `--bf16`オプションでbf16（bfloat16）での推論を行うこともできます。

`--interactive`オプションで対話モードを指定しています。

Stable Diffusion 2.0（またはそこからの追加学習モデル）を使う場合は`--v2`オプションを追加してください。v-parameterizationを使うモデル（`768-v-ema.ckpt`およびそこからの追加学習モデル）を使う場合はさらに`--v_parameterization`を追加してください。

`--v2`の指定有無が間違っているとモデル読み込み時にエラーになります。`--v_parameterization`の指定有無が間違っていると茶色い画像が表示されます。

`Type prompt:`と表示されたらプロンプトを入力してください。

![image](https://user-images.githubusercontent.com/52813779/235343115-f3b8ac82-456d-4aab-9724-0cc73c4534aa.png)

※画像が表示されずエラーになる場合、headless（画面表示機能なし）のOpenCVがインストールされているかもしれません。`pip install opencv-python`として通常のOpenCVを入れてください。または`--no_preview`オプションで画像表示を止めてください。

画像ウィンドウを選択してから何らかのキーを押すとウィンドウが閉じ、次のプロンプトが入力できます。プロンプトでCtrl+Z、エンターの順に打鍵するとスクリプトを閉じます。

## 単一のプロンプトで画像を一括生成

以下のように入力します（実際には1行で入力します）。

```batchfile
python gen_img_diffusers.py --ckpt <モデル名> --outdir <画像出力先> 
    --xformers --fp16 --images_per_prompt <生成枚数> --prompt "<プロンプト>"
```

`--images_per_prompt`オプションで、プロンプト1件当たりの生成枚数を指定します。`--prompt`オプションでプロンプトを指定します。スペースを含む場合はダブルクォーテーションで囲んでください。

`--batch_size`オプションでバッチサイズを指定できます（後述）。

## ファイルからプロンプトを読み込み一括生成

以下のように入力します。

```batchfile
python gen_img_diffusers.py --ckpt <モデル名> --outdir <画像出力先> 
    --xformers --fp16 --from_file <プロンプトファイル名>
```

`--from_file`オプションで、プロンプトが記述されたファイルを指定します。1行1プロンプトで記述してください。`--images_per_prompt`オプションを指定して1行あたり生成枚数を指定できます。

## ネガティブプロンプト、重みづけの使用

プロンプトオプション（プロンプト内で`--x`のように指定、後述）で`--n`を書くと、以降がネガティブプロンプトとなります。

またAUTOMATIC1111氏のWeb UIと同様の `()` や` []` 、`(xxx:1.3)` などによる重みづけが可能です（実装はDiffusersの[Long Prompt Weighting Stable Diffusion](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#long-prompt-weighting-stable-diffusion)からコピーしたものです）。

コマンドラインからのプロンプト指定、ファイルからのプロンプト読み込みでも同様に指定できます。

![image](https://user-images.githubusercontent.com/52813779/235343128-e79cd768-ec59-46f5-8395-fce9bdc46208.png)

# 主なオプション

コマンドラインから指定してください。

## モデルの指定

- `--ckpt <モデル名>`：モデル名を指定します。`--ckpt`オプションは必須です。Stable Diffusionのcheckpointファイル、またはDiffusersのモデルフォルダ、Hugging FaceのモデルIDを指定できます。

- `--v2`：Stable Diffusion 2.x系のモデルを使う場合に指定します。1.x系の場合には指定不要です。

- `--v_parameterization`：v-parameterizationを使うモデルを使う場合に指定します（`768-v-ema.ckpt`およびそこからの追加学習モデル、Waifu Diffusion v1.5など）。
    
    `--v2`の指定有無が間違っているとモデル読み込み時にエラーになります。`--v_parameterization`の指定有無が間違っていると茶色い画像が表示されます。

- `--vae`：使用するVAEを指定します。未指定時はモデル内のVAEを使用します。

## 画像生成と出力

- `--interactive`：インタラクティブモードで動作します。プロンプトを入力すると画像が生成されます。

- `--prompt <プロンプト>`：プロンプトを指定します。スペースを含む場合はダブルクォーテーションで囲んでください。

- `--from_file <プロンプトファイル名>`：プロンプトが記述されたファイルを指定します。1行1プロンプトで記述してください。なお画像サイズやguidance scaleはプロンプトオプション（後述）で指定できます。

- `--W <画像幅>`：画像の幅を指定します。デフォルトは`512`です。

- `--H <画像高さ>`：画像の高さを指定します。デフォルトは`512`です。

- `--steps <ステップ数>`：サンプリングステップ数を指定します。デフォルトは`50`です。

- `--scale <ガイダンススケール>`：unconditionalガイダンススケールを指定します。デフォルトは`7.5`です。

- `--sampler <サンプラー名>`：サンプラーを指定します。デフォルトは`ddim`です。Diffusersで提供されているddim、pndm、dpmsolver、dpmsolver+++、lms、euler、euler_a、が指定可能です（後ろの三つはk_lms、k_euler、k_euler_aでも指定できます）。

- `--outdir <画像出力先フォルダ>`：画像の出力先を指定します。

- `--images_per_prompt <生成枚数>`：プロンプト1件当たりの生成枚数を指定します。デフォルトは`1`です。

- `--clip_skip <スキップ数>`：CLIPの後ろから何番目の層を使うかを指定します。省略時は最後の層を使います。

- `--max_embeddings_multiples <倍数>`：CLIPの入出力長をデフォルト（75）の何倍にするかを指定します。未指定時は75のままです。たとえば3を指定すると入出力長が225になります。

- `--negative_scale` : uncoditioningのguidance scaleを個別に指定します。[gcem156氏のこちらの記事](https://note.com/gcem156/n/ne9a53e4a6f43)を参考に実装したものです。

## メモリ使用量や生成速度の調整

- `--batch_size <バッチサイズ>`：バッチサイズを指定します。デフォルトは`1`です。バッチサイズが大きいとメモリを多く消費しますが、生成速度が速くなります。

- `--vae_batch_size <VAEのバッチサイズ>`：VAEのバッチサイズを指定します。デフォルトはバッチサイズと同じです。
    VAEのほうがメモリを多く消費するため、デノイジング後（stepが100%になった後）でメモリ不足になる場合があります。このような場合にはVAEのバッチサイズを小さくしてください。

- `--xformers`：xformersを使う場合に指定します。

- `--fp16`：fp16（単精度）での推論を行います。`fp16`と`bf16`をどちらも指定しない場合はfp32（単精度）での推論を行います。

- `--bf16`：bf16（bfloat16）での推論を行います。RTX 30系のGPUでのみ指定可能です。`--bf16`オプションはRTX 30系以外のGPUではエラーになります。`fp16`よりも`bf16`のほうが推論結果がNaNになる（真っ黒の画像になる）可能性が低いようです。

## 追加ネットワーク（LoRA等）の使用

- `--network_module`：使用する追加ネットワークを指定します。LoRAの場合は`--network_module networks.lora`と指定します。複数のLoRAを使用する場合は`--network_module networks.lora networks.lora networks.lora`のように指定します。

- `--network_weights`：使用する追加ネットワークの重みファイルを指定します。`--network_weights model.safetensors`のように指定します。複数のLoRAを使用する場合は`--network_weights model1.safetensors model2.safetensors model3.safetensors`のように指定します。引数の数は`--network_module`で指定した数と同じにしてください。

- `--network_mul`：使用する追加ネットワークの重みを何倍にするかを指定します。デフォルトは`1`です。`--network_mul 0.8`のように指定します。複数のLoRAを使用する場合は`--network_mul 0.4 0.5 0.7`のように指定します。引数の数は`--network_module`で指定した数と同じにしてください。

- `--network_merge`：使用する追加ネットワークの重みを`--network_mul`に指定した重みであらかじめマージします。`--network_pre_calc` と同時に使用できません。プロンプトオプションの`--am`、およびRegional LoRAは使用できなくなりますが、LoRA未使用時と同じ程度まで生成が高速化されます。

- `--network_pre_calc`：使用する追加ネットワークの重みを生成ごとにあらかじめ計算します。プロンプトオプションの`--am`が使用できます。LoRA未使用時と同じ程度まで生成は高速化されますが、生成前に重みを計算する時間が必要で、またメモリ使用量も若干増加します。Regional LoRA使用時は無効になります 。

# 主なオプションの指定例

次は同一プロンプトで64枚をバッチサイズ4で一括生成する例です。

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 64 
    --prompt "beautiful flowers --n monochrome"
```

次はファイルに書かれたプロンプトを、それぞれ10枚ずつ、バッチサイズ4で一括生成する例です。

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 10 
    --from_file prompts.txt
```

Textual Inversion（後述）およびLoRAの使用例です。

```batchfile
python gen_img_diffusers.py --ckpt model.safetensors 
    --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --fp16 --sampler k_euler_a 
    --textual_inversion_embeddings goodembed.safetensors negprompt.pt 
    --network_module networks.lora networks.lora 
    --network_weights model1.safetensors model2.safetensors 
    --network_mul 0.4 0.8 
    --clip_skip 2 --max_embeddings_multiples 1 
    --batch_size 8 --images_per_prompt 1 --interactive
```

# プロンプトオプション

プロンプト内で、`--n`のように「ハイフンふたつ+アルファベットn文字」でプロンプトから各種オプションの指定が可能です。対話モード、コマンドライン、ファイル、いずれからプロンプトを指定する場合でも有効です。

プロンプトのオプション指定`--n`の前後にはスペースを入れてください。

- `--n`：ネガティブプロンプトを指定します。

- `--w`：画像幅を指定します。コマンドラインからの指定を上書きします。

- `--h`：画像高さを指定します。コマンドラインからの指定を上書きします。

- `--s`：ステップ数を指定します。コマンドラインからの指定を上書きします。

- `--d`：この画像の乱数seedを指定します。`--images_per_prompt`を指定している場合は「--d 1,2,3,4」のようにカンマ区切りで複数指定してください。
    ※様々な理由により、Web UIとは同じ乱数seedでも生成される画像が異なる場合があります。

- `--l`：guidance scaleを指定します。コマンドラインからの指定を上書きします。

- `--t`：img2img（後述）のstrengthを指定します。コマンドラインからの指定を上書きします。

- `--nl`：ネガティブプロンプトのguidance scaleを指定します（後述）。コマンドラインからの指定を上書きします。

- `--am`：追加ネットワークの重みを指定します。コマンドラインからの指定を上書きします。複数の追加ネットワークを使用する場合は`--am 0.8,0.5,0.3`のように __カンマ区切りで__ 指定します。

※これらのオプションを指定すると、バッチサイズよりも小さいサイズでバッチが実行される場合があります（これらの値が異なると一括生成できないため）。（あまり気にしなくて大丈夫ですが、ファイルからプロンプトを読み込み生成する場合は、これらの値が同一のプロンプトを並べておくと効率が良くなります。）

例：
```
(masterpiece, best quality), 1girl, in shirt and plated skirt, standing at street under cherry blossoms, upper body, [from below], kind smile, looking at another, [goodembed] --n realistic, real life, (negprompt), (lowres:1.1), (worst quality:1.2), (low quality:1.1), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry --w 960 --h 640 --s 28 --d 1
```

![image](https://user-images.githubusercontent.com/52813779/235343446-25654172-fff4-4aaf-977a-20d262b51676.png)

# img2img

## オプション

- `--image_path`：img2imgに利用する画像を指定します。`--image_path template.png`のように指定します。フォルダを指定すると、そのフォルダの画像を順次利用します。

- `--strength`：img2imgのstrengthを指定します。`--strength 0.8`のように指定します。デフォルトは`0.8`です。

- `--sequential_file_name`：ファイル名を連番にするかどうかを指定します。指定すると生成されるファイル名が`im_000001.png`からの連番になります。

- `--use_original_file_name`：指定すると生成ファイル名がオリジナルのファイル名と同じになります。

## コマンドラインからの実行例

```batchfile
python gen_img_diffusers.py --ckpt trinart_characters_it4_v1_vae_merged.ckpt 
    --outdir outputs --xformers --fp16 --scale 12.5 --sampler k_euler --steps 32 
    --image_path template.png --strength 0.8 
    --prompt "1girl, cowboy shot, brown hair, pony tail, brown eyes, 
          sailor school uniform, outdoors 
          --n lowres, bad anatomy, bad hands, error, missing fingers, cropped, 
          worst quality, low quality, normal quality, jpeg artifacts, (blurry), 
          hair ornament, glasses" 
    --batch_size 8 --images_per_prompt 32
```

`--image_path`オプションにフォルダを指定すると、そのフォルダの画像を順次読み込みます。生成される枚数は画像枚数ではなく、プロンプト数になりますので、`--images_per_promptPPオプションを指定してimg2imgする画像の枚数とプロンプト数を合わせてください。

ファイルはファイル名でソートして読み込みます。なおソート順は文字列順となりますので（`1.jpg→2.jpg→10.jpg`ではなく`1.jpg→10.jpg→2.jpg`の順）、頭を0埋めするなどしてご対応ください（`01.jpg→02.jpg→10.jpg`）。

## img2imgを利用したupscale

img2img時にコマンドラインオプションの`--W`と`--H`で生成画像サイズを指定すると、元画像をそのサイズにリサイズしてからimg2imgを行います。

またimg2imgの元画像がこのスクリプトで生成した画像の場合、プロンプトを省略すると、元画像のメタデータからプロンプトを取得しそのまま用います。これによりHighres. fixの2nd stageの動作だけを行うことができます。

## img2img時のinpainting

画像およびマスク画像を指定してinpaintingできます（inpaintingモデルには対応しておらず、単にマスク領域を対象にimg2imgするだけです）。

オプションは以下の通りです。

- `--mask_image`：マスク画像を指定します。`--img_path`と同様にフォルダを指定すると、そのフォルダの画像を順次利用します。

マスク画像はグレースケール画像で、白の部分がinpaintingされます。境界をグラデーションしておくとなんとなく滑らかになりますのでお勧めです。

![image](https://user-images.githubusercontent.com/52813779/235343795-9eaa6d98-02ff-4f32-b089-80d1fc482453.png)

# その他の機能

## Textual Inversion

`--textual_inversion_embeddings`オプションで使用するembeddingsを指定します（複数指定可）。拡張子を除いたファイル名をプロンプト内で使用することで、そのembeddingsを利用します（Web UIと同様の使用法です）。ネガティブプロンプト内でも使用できます。

モデルとして、当リポジトリで学習したTextual Inversionモデル、およびWeb UIで学習したTextual Inversionモデル（画像埋め込みは非対応）を利用できます

## Extended Textual Inversion

`--textual_inversion_embeddings`の代わりに`--XTI_embeddings`オプションを指定してください。使用法は`--textual_inversion_embeddings`と同じです。

## Highres. fix

AUTOMATIC1111氏のWeb UIにある機能の類似機能です（独自実装のためもしかしたらいろいろ異なるかもしれません）。最初に小さめの画像を生成し、その画像を元にimg2imgすることで、画像全体の破綻を防ぎつつ大きな解像度の画像を生成します。

2nd stageのstep数は`--steps` と`--strength`オプションの値から計算されます（`steps*strength`）。

img2imgと併用できません。

以下のオプションがあります。

- `--highres_fix_scale`：Highres. fixを有効にして、1st stageで生成する画像のサイズを、倍率で指定します。最終出力が1024x1024で、最初に512x512の画像を生成する場合は`--highres_fix_scale 0.5`のように指定します。Web UI出の指定の逆数になっていますのでご注意ください。

- `--highres_fix_steps`：1st stageの画像のステップ数を指定します。デフォルトは`28`です。

- `--highres_fix_save_1st`：1st stageの画像を保存するかどうかを指定します。

- `--highres_fix_latents_upscaling`：指定すると2nd stageの画像生成時に1st stageの画像をlatentベースでupscalingします（bilinearのみ対応）。未指定時は画像をLANCZOS4でupscalingします。

- `--highres_fix_upscaler`：2nd stageに任意のupscalerを利用します。現在は`--highres_fix_upscaler tools.latent_upscaler` のみ対応しています。

- `--highres_fix_upscaler_args`：`--highres_fix_upscaler`で指定したupscalerに渡す引数を指定します。
    `tools.latent_upscaler`の場合は、`--highres_fix_upscaler_args "weights=D:\Work\SD\Models\others\etc\upscaler-v1-e100-220.safetensors"`のように重みファイルを指定します。 

コマンドラインの例です。

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 1024 --H 1024 --batch_size 1 --outdir ../txt2img 
    --steps 48 --sampler ddim --fp16 
    --xformers 
    --images_per_prompt 1  --interactive 
    --highres_fix_scale 0.5 --highres_fix_steps 28 --strength 0.5
```

## ControlNet

現在はControlNet 1.0のみ動作確認しています。プリプロセスはCannyのみサポートしています。

以下のオプションがあります。

- `--control_net_models`：ControlNetのモデルファイルを指定します。
    複数指定すると、それらをstepごとに切り替えて利用します（Web UIのControlNet拡張の実装と異なります）。diffと通常の両方をサポートします。

- `--guide_image_path`：ControlNetに使うヒント画像を指定します。`--img_path`と同様にフォルダを指定すると、そのフォルダの画像を順次利用します。Canny以外のモデルの場合には、あらかじめプリプロセスを行っておいてください。

- `--control_net_preps`：ControlNetのプリプロセスを指定します。`--control_net_models`と同様に複数指定可能です。現在はcannyのみ対応しています。対象モデルでプリプロセスを使用しない場合は `none` を指定します。
   cannyの場合 `--control_net_preps canny_63_191`のように、閾値1と2を'_'で区切って指定できます。

- `--control_net_weights`：ControlNetの適用時の重みを指定します（`1.0`で通常、`0.5`なら半分の影響力で適用）。`--control_net_models`と同様に複数指定可能です。

- `--control_net_ratios`：ControlNetを適用するstepの範囲を指定します。`0.5`の場合は、step数の半分までControlNetを適用します。`--control_net_models`と同様に複数指定可能です。

コマンドラインの例です。

```batchfile
python gen_img_diffusers.py --ckpt model_ckpt --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --bf16 --sampler k_euler_a 
    --control_net_models diff_control_sd15_canny.safetensors --control_net_weights 1.0 
    --guide_image_path guide.png --control_net_ratios 1.0 --interactive
```

## Attention Couple + Reginal LoRA

プロンプトをいくつかの部分に分割し、それぞれのプロンプトを画像内のどの領域に適用するかを指定できる機能です。個別のオプションはありませんが、`mask_path`とプロンプトで指定します。

まず、プロンプトで` AND `を利用して、複数部分を定義します。最初の3つに対して領域指定ができ、以降の部分は画像全体へ適用されます。ネガティブプロンプトは画像全体に適用されます。

以下ではANDで3つの部分を定義しています。

```
shs 2girls, looking at viewer, smile AND bsb 2girls, looking back AND 2girls --n bad quality, worst quality
```

次にマスク画像を用意します。マスク画像はカラーの画像で、RGBの各チャネルがプロンプトのANDで区切られた部分に対応します。またあるチャネルの値がすべて0の場合、画像全体に適用されます。

上記の例では、Rチャネルが`shs 2girls, looking at viewer, smile`、Gチャネルが`bsb 2girls, looking back`に、Bチャネルが`2girls`に対応します。次のようなマスク画像を使用すると、Bチャネルに指定がありませんので、`2girls`は画像全体に適用されます。

![image](https://user-images.githubusercontent.com/52813779/235343061-b4dc9392-3dae-4831-8347-1e9ae5054251.png)

マスク画像は`--mask_path`で指定します。現在は1枚のみ対応しています。指定した画像サイズに自動的にリサイズされ適用されます。

ControlNetと組み合わせることも可能です（細かい位置指定にはControlNetとの組み合わせを推奨します）。

LoRAを指定すると、`--network_weights`で指定した複数のLoRAがそれぞれANDの各部分に対応します。現在の制約として、LoRAの数はANDの部分の数と同じである必要があります。

## CLIP Guided Stable Diffusion

DiffusersのCommunity Examplesの[こちらのcustom pipeline](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#clip-guided-stable-diffusion)からソースをコピー、変更したものです。

通常のプロンプトによる生成指定に加えて、追加でより大規模のCLIPでプロンプトのテキストの特徴量を取得し、生成中の画像の特徴量がそのテキストの特徴量に近づくよう、生成される画像をコントロールします（私のざっくりとした理解です）。大きめのCLIPを使いますのでVRAM使用量はかなり増加し（VRAM 8GBでは512*512でも厳しいかもしれません）、生成時間も掛かります。

なお選択できるサンプラーはDDIM、PNDM、LMSのみとなります。

`--clip_guidance_scale`オプションにどの程度、CLIPの特徴量を反映するかを数値で指定します。先のサンプルでは100になっていますので、そのあたりから始めて増減すると良いようです。

デフォルトではプロンプトの先頭75トークン（重みづけの特殊文字を除く）がCLIPに渡されます。プロンプトの`--c`オプションで、通常のプロンプトではなく、CLIPに渡すテキストを別に指定できます（たとえばCLIPはDreamBoothのidentifier（識別子）や「1girl」などのモデル特有の単語は認識できないと思われますので、それらを省いたテキストが良いと思われます）。

コマンドラインの例です。

```batchfile
python gen_img_diffusers.py  --ckpt v1-5-pruned-emaonly.ckpt --n_iter 1 
    --scale 2.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img --steps 36  
    --sampler ddim --fp16 --opt_channels_last --xformers --images_per_prompt 1  
    --interactive --clip_guidance_scale 100
```

## CLIP Image Guided Stable Diffusion

テキストではなくCLIPに別の画像を渡し、その特徴量に近づくよう生成をコントロールする機能です。`--clip_image_guidance_scale`オプションで適用量の数値を、`--guide_image_path`オプションでguideに使用する画像（ファイルまたはフォルダ）を指定してください。

コマンドラインの例です。

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img 
    --steps 80 --sampler ddim --fp16 --opt_channels_last --xformers 
    --images_per_prompt 1  --interactive  --clip_image_guidance_scale 100 
    --guide_image_path YUKA160113420I9A4104_TP_V.jpg
```

### VGG16 Guided Stable Diffusion

指定した画像に近づくように画像生成する機能です。通常のプロンプトによる生成指定に加えて、追加でVGG16の特徴量を取得し、生成中の画像が指定したガイド画像に近づくよう、生成される画像をコントロールします。img2imgでの使用をお勧めします（通常の生成では画像がぼやけた感じになります）。CLIP Guided Stable Diffusionの仕組みを流用した独自の機能です。またアイデアはVGGを利用したスタイル変換から拝借しています。

なお選択できるサンプラーはDDIM、PNDM、LMSのみとなります。

`--vgg16_guidance_scale`オプションにどの程度、VGG16特徴量を反映するかを数値で指定します。試した感じでは100くらいから始めて増減すると良いようです。`--guide_image_path`オプションでguideに使用する画像（ファイルまたはフォルダ）を指定してください。

複数枚の画像を一括でimg2img変換し、元画像をガイド画像とする場合、`--guide_image_path`と`--image_path`に同じ値を指定すればOKです。

コマンドラインの例です。

```batchfile
python gen_img_diffusers.py --ckpt wd-v1-3-full-pruned-half.ckpt 
    --n_iter 1 --scale 5.5 --steps 60 --outdir ../txt2img 
    --xformers --sampler ddim --fp16 --W 512 --H 704 
    --batch_size 1 --images_per_prompt 1 
    --prompt "picturesque, 1girl, solo, anime face, skirt, beautiful face 
        --n lowres, bad anatomy, bad hands, error, missing fingers, 
        cropped, worst quality, low quality, normal quality, 
        jpeg artifacts, blurry, 3d, bad face, monochrome --d 1" 
    --strength 0.8 --image_path ..\src_image
    --vgg16_guidance_scale 100 --guide_image_path ..\src_image 
```

`--vgg16_guidance_layerPで特徴量取得に使用するVGG16のレイヤー番号を指定できます（デフォルトは20でconv4-2のReLUです）。上の層ほど画風を表現し、下の層ほどコンテンツを表現するといわれています。

![image](https://user-images.githubusercontent.com/52813779/235343813-3c1f0d7a-4fb3-4274-98e4-b92d76b551df.png)

# その他のオプション

- `--no_preview` : 対話モードでプレビュー画像を表示しません。OpenCVがインストールされていない場合や、出力されたファイルを直接確認する場合に指定してください。

- `--n_iter` : 生成を繰り返す回数を指定します。デフォルトは1です。プロンプトをファイルから読み込むとき、複数回の生成を行いたい場合に指定します。

- `--tokenizer_cache_dir` : トークナイザーのキャッシュディレクトリを指定します。（作業中）

- `--seed` : 乱数seedを指定します。1枚生成時はその画像のseed、複数枚生成時は各画像のseedを生成するための乱数のseedになります（`--from_file`で複数画像生成するとき、`--seed`オプションを指定すると複数回実行したときに各画像が同じseedになります）。

- `--iter_same_seed` : プロンプトに乱数seedの指定がないとき、`--n_iter`の繰り返し内ではすべて同じseedを使います。`--from_file`で指定した複数のプロンプト間でseedを統一して比較するときに使います。

- `--diffusers_xformers` : Diffuserのxformersを使用します。

- `--opt_channels_last` : 推論時にテンソルのチャンネルを最後に配置します。場合によっては高速化されることがあります。

- `--network_show_meta` : 追加ネットワークのメタデータを表示します。

