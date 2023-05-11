For non-Japanese speakers: this README is provided only in Japanese in the current state. Sorry for inconvenience. We will provide English version in the near future.

`--dataset_config` で渡すことができる設定ファイルに関する説明です。

## 概要

設定ファイルを渡すことにより、ユーザが細かい設定を行えるようにします。

* 複数のデータセットが設定可能になります
    * 例えば `resolution` をデータセットごとに設定して、それらを混合して学習できます。
    * DreamBooth の手法と fine tuning の手法の両方に対応している学習方法では、DreamBooth 方式と fine tuning 方式のデータセットを混合することが可能です。
* サブセットごとに設定を変更することが可能になります
    * データセットを画像ディレクトリ別またはメタデータ別に分割したものがサブセットです。いくつかのサブセットが集まってデータセットを構成します。
    * `keep_tokens` や `flip_aug` 等のオプションはサブセットごとに設定可能です。一方、`resolution` や `batch_size` といったオプションはデータセットごとに設定可能で、同じデータセットに属するサブセットでは値が共通になります。詳しくは後述します。

設定ファイルの形式は JSON か TOML を利用できます。記述のしやすさを考えると [TOML](https://toml.io/ja/v1.0.0-rc.2) を利用するのがオススメです。以下、TOML の利用を前提に説明します。

TOML で記述した設定ファイルの例です。

```toml
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

# これは DreamBooth 方式のデータセット
[[datasets]]
resolution = 512
batch_size = 4
keep_tokens = 2

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
  class_tokens = 'hoge girl'
  # このサブセットは keep_tokens = 2 （所属する datasets の値が使われる）

  [[datasets.subsets]]
  image_dir = 'C:\fuga'
  class_tokens = 'fuga boy'
  keep_tokens = 3

  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'
  class_tokens = 'human'
  keep_tokens = 1

# これは fine tuning 方式のデータセット
[[datasets]]
resolution = [768, 768]
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'C:\piyo'
  metadata_file = 'C:\piyo\piyo_md.json'
  # このサブセットは keep_tokens = 1 （general の値が使われる）
```

この例では、3 つのディレクトリを DreamBooth 方式のデータセットとして 512x512 (batch size 4) で学習させ、1 つのディレクトリを fine tuning 方式のデータセットとして 768x768 (batch size 2) で学習させることになります。

## データセット・サブセットに関する設定

データセット・サブセットに関する設定は、登録可能な箇所がいくつかに分かれています。

* `[general]`
    * 全データセットまたは全サブセットに適用されるオプションを指定する箇所です。
    * データセットごとの設定及びサブセットごとの設定に同名のオプションが存在していた場合には、データセット・サブセットごとの設定が優先されます。
* `[[datasets]]`
    * `datasets` はデータセットに関する設定の登録箇所になります。各データセットに個別に適用されるオプションを指定する箇所です。
    * サブセットごとの設定が存在していた場合には、サブセットごとの設定が優先されます。
* `[[datasets.subsets]]`
    * `datasets.subsets` はサブセットに関する設定の登録箇所になります。各サブセットに個別に適用されるオプションを指定する箇所です。

先程の例における、画像ディレクトリと登録箇所の対応に関するイメージ図です。

```
C:\
├─ hoge  ->  [[datasets.subsets]] No.1  ┐                        ┐
├─ fuga  ->  [[datasets.subsets]] No.2  |->  [[datasets]] No.1   |->  [general]
├─ reg   ->  [[datasets.subsets]] No.3  ┘                        |
└─ piyo  ->  [[datasets.subsets]] No.4  -->  [[datasets]] No.2   ┘
```

画像ディレクトリがそれぞれ1つの `[[datasets.subsets]]` に対応しています。そして `[[datasets.subsets]]` が1つ以上組み合わさって1つの `[[datasets]]` を構成します。`[general]` には全ての `[[datasets]]`, `[[datasets.subsets]]` が属します。

登録箇所ごとに指定可能なオプションは異なりますが、同名のオプションが指定された場合は下位の登録箇所にある値が優先されます。先程の例の `keep_tokens` オプションの扱われ方を確認してもらうと理解しやすいかと思います。

加えて、学習方法が対応している手法によっても指定可能なオプションが変化します。

* DreamBooth 方式専用のオプション
* fine tuning 方式専用のオプション
* caption dropout の手法が使える場合のオプション

DreamBooth の手法と fine tuning の手法の両方とも利用可能な学習方法では、両者を併用することができます。
併用する際の注意点として、DreamBooth 方式なのか fine tuning 方式なのかはデータセット単位で判別を行っているため、同じデータセット中に DreamBooth 方式のサブセットと fine tuning 方式のサブセットを混在させることはできません。
つまり、これらを併用したい場合には異なる方式のサブセットが異なるデータセットに所属するように設定する必要があります。

プログラムの挙動としては、後述する `metadata_file` オプションが存在していたら fine tuning 方式のサブセットだと判断します。
そのため、同一のデータセットに所属するサブセットについて言うと、「全てが `metadata_file` オプションを持つ」か「全てが `metadata_file` オプションを持たない」かのどちらかになっていれば問題ありません。

以下、利用可能なオプションを説明します。コマンドライン引数と名称が同一のオプションについては、基本的に説明を割愛します。他の README を参照してください。

### 全学習方法で共通のオプション

学習方法によらずに指定可能なオプションです。

#### データセット向けオプション

データセットの設定に関わるオプションです。`datasets.subsets` には記述できません。

| オプション名 | 設定例 | `[general]` | `[[datasets]]` |
| ---- | ---- | ---- | ---- |
| `batch_size` | `1` | o | o |
| `bucket_no_upscale` | `true` | o | o |
| `bucket_reso_steps` | `64` | o | o |
| `enable_bucket` | `true` | o | o |
| `max_bucket_reso` | `1024` | o | o |
| `min_bucket_reso` | `128` | o | o |
| `resolution` | `256`, `[512, 512]` | o | o |

* `batch_size`
    * コマンドライン引数の `--train_batch_size` と同等です。

これらの設定はデータセットごとに固定です。
つまり、データセットに所属するサブセットはこれらの設定を共有することになります。
例えば解像度が異なるデータセットを用意したい場合は、上に挙げた例のように別々のデータセットとして定義すれば別々の解像度を設定可能です。

#### サブセット向けオプション

サブセットの設定に関わるオプションです。

| オプション名 | 設定例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `color_aug` | `false` | o | o | o |
| `face_crop_aug_range` | `[1.0, 3.0]` | o | o | o |
| `flip_aug` | `true` | o | o | o |
| `keep_tokens` | `2` | o | o | o |
| `num_repeats` | `10` | o | o | o |
| `random_crop` | `false` | o | o | o |
| `shuffle_caption` | `true` | o | o | o |

* `num_repeats`
    * サブセットの画像の繰り返し回数を指定します。fine tuning における `--dataset_repeats` に相当しますが、`num_repeats` はどの学習方法でも指定可能です。

### DreamBooth 方式専用のオプション

DreamBooth 方式のオプションは、サブセット向けオプションのみ存在します。

#### サブセット向けオプション

DreamBooth 方式のサブセットの設定に関わるオプションです。

| オプション名 | 設定例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o（必須） |
| `caption_extension` | `".txt"` | o | o | o |
| `class_tokens` | `“sks girl”` | - | - | o |
| `is_reg` | `false` | - | - | o |

まず注意点として、 `image_dir` には画像ファイルが直下に置かれているパスを指定する必要があります。従来の DreamBooth の手法ではサブディレクトリに画像を置く必要がありましたが、そちらとは仕様に互換性がありません。また、`5_cat` のようなフォルダ名にしても、画像の繰り返し回数とクラス名は反映されません。これらを個別に設定したい場合、`num_repeats` と `class_tokens` で明示的に指定する必要があることに注意してください。

* `image_dir`
    * 画像ディレクトリのパスを指定します。指定必須オプションです。
    * 画像はディレクトリ直下に置かれている必要があります。
* `class_tokens`
    * クラストークンを設定します。
    * 画像に対応する caption ファイルが存在しない場合にのみ学習時に利用されます。利用するかどうかの判定は画像ごとに行います。`class_tokens` を指定しなかった場合に caption ファイルも見つからなかった場合にはエラーになります。
* `is_reg`
    * サブセットの画像が正規化用かどうかを指定します。指定しなかった場合は `false` として、つまり正規化画像ではないとして扱います。

### fine tuning 方式専用のオプション

fine tuning 方式のオプションは、サブセット向けオプションのみ存在します。

#### サブセット向けオプション

fine tuning 方式のサブセットの設定に関わるオプションです。

| オプション名 | 設定例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o |
| `metadata_file` | `'C:\piyo\piyo_md.json'` | - | - | o（必須） |

* `image_dir`
    * 画像ディレクトリのパスを指定します。DreamBooth の手法の方とは異なり指定は必須ではありませんが、設定することを推奨します。
        * 指定する必要がない状況としては、メタデータファイルの生成時に `--full_path` を付与して実行していた場合です。
    * 画像はディレクトリ直下に置かれている必要があります。
* `metadata_file`
    * サブセットで利用されるメタデータファイルのパスを指定します。指定必須オプションです。
        * コマンドライン引数の `--in_json` と同等です。
    * サブセットごとにメタデータファイルを指定する必要がある仕様上、ディレクトリを跨いだメタデータを1つのメタデータファイルとして作成することは避けた方が良いでしょう。画像ディレクトリごとにメタデータファイルを用意し、それらを別々のサブセットとして登録することを強く推奨します。

### caption dropout の手法が使える場合に指定可能なオプション

caption dropout の手法が使える場合のオプションは、サブセット向けオプションのみ存在します。
DreamBooth 方式か fine tuning 方式かに関わらず、caption dropout に対応している学習方法であれば指定可能です。

#### サブセット向けオプション

caption dropout が使えるサブセットの設定に関わるオプションです。

| オプション名 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- |
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |

## 重複したサブセットが存在する時の挙動

DreamBooth 方式のデータセットの場合、その中にある `image_dir` が同一のサブセットは重複していると見なされます。
fine tuning 方式のデータセットの場合は、その中にある `metadata_file` が同一のサブセットは重複していると見なされます。
データセット中に重複したサブセットが存在する場合、2個目以降は無視されます。

一方、異なるデータセットに所属している場合は、重複しているとは見なされません。
例えば、以下のように同一の `image_dir` を持つサブセットを別々のデータセットに入れた場合には、重複していないと見なします。
これは、同じ画像でも異なる解像度で学習したい場合に役立ちます。

```toml
# 別々のデータセットに存在している場合は重複とは見なされず、両方とも学習に使われる

[[datasets]]
resolution = 512

  [[datasets.subsets]]
  image_dir = 'C:\hoge'

[[datasets]]
resolution = 768

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
```

## コマンドライン引数との併用

設定ファイルのオプションの中には、コマンドライン引数のオプションと役割が重複しているものがあります。

以下に挙げるコマンドライン引数のオプションは、設定ファイルを渡した場合には無視されます。

* `--train_data_dir`
* `--reg_data_dir`
* `--in_json`

以下に挙げるコマンドライン引数のオプションは、コマンドライン引数と設定ファイルで同時に指定された場合、コマンドライン引数の値よりも設定ファイルの値が優先されます。特に断りがなければ同名のオプションとなります。

| コマンドライン引数のオプション     | 優先される設定ファイルのオプション |
| ---------------------------------- | ---------------------------------- |
| `--bucket_no_upscale`              |                                    |
| `--bucket_reso_steps`              |                                    |
| `--caption_dropout_every_n_epochs` |                                    |
| `--caption_dropout_rate`           |                                    |
| `--caption_extension`              |                                    |
| `--caption_tag_dropout_rate`       |                                    |
| `--color_aug`                      |                                    |
| `--dataset_repeats`                | `num_repeats`                      |
| `--enable_bucket`                  |                                    |
| `--face_crop_aug_range`            |                                    |
| `--flip_aug`                       |                                    |
| `--keep_tokens`                    |                                    |
| `--min_bucket_reso`                |                                    |
| `--random_crop`                    |                                    |
| `--resolution`                     |                                    |
| `--shuffle_caption`                |                                    |
| `--train_batch_size`               | `batch_size`                       |

## エラーの手引き

現在、外部ライブラリを利用して設定ファイルの記述が正しいかどうかをチェックしているのですが、整備が行き届いておらずエラーメッセージがわかりづらいという問題があります。
将来的にはこの問題の改善に取り組む予定です。

次善策として、頻出のエラーとその対処法について載せておきます。
正しいはずなのにエラーが出る場合、エラー内容がどうしても分からない場合は、バグかもしれないのでご連絡ください。

* `voluptuous.error.MultipleInvalid: required key not provided @ ...`: 指定必須のオプションが指定されていないというエラーです。指定を忘れているか、オプション名を間違って記述している可能性が高いです。
  * `...` の箇所にはエラーが発生した場所が載っています。例えば `voluptuous.error.MultipleInvalid: required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']` のようなエラーが出たら、0 番目の `datasets` 中の 0 番目の `subsets` の設定に `image_dir` が存在しないということになります。
* `voluptuous.error.MultipleInvalid: expected int for dictionary value @ ...`: 指定する値の形式が不正というエラーです。値の形式が間違っている可能性が高いです。`int` の部分は対象となるオプションによって変わります。この README に載っているオプションの「設定例」が役立つかもしれません。
* `voluptuous.error.MultipleInvalid: extra keys not allowed @ ...`: 対応していないオプション名が存在している場合に発生するエラーです。オプション名を間違って記述しているか、誤って紛れ込んでいる可能性が高いです。


