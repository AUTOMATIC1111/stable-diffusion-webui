import argparse
from dataclasses import (
  asdict,
  dataclass,
)
import functools
import random
from textwrap import dedent, indent
import json
from pathlib import Path
# from toolz import curry
from typing import (
  List,
  Optional,
  Sequence,
  Tuple,
  Union,
)

import toml
import voluptuous
from voluptuous import (
  Any,
  ExactSequence,
  MultipleInvalid,
  Object,
  Required,
  Schema,
)
from transformers import CLIPTokenizer

from . import train_util
from .train_util import (
  DreamBoothSubset,
  FineTuningSubset,
  DreamBoothDataset,
  FineTuningDataset,
  DatasetGroup,
)


def add_config_arguments(parser: argparse.ArgumentParser):
  parser.add_argument("--dataset_config", type=Path, default=None, help="config file for detail settings / 詳細な設定用の設定ファイル")

# TODO: inherit Params class in Subset, Dataset

@dataclass
class BaseSubsetParams:
  image_dir: Optional[str] = None
  num_repeats: int = 1
  shuffle_caption: bool = False
  keep_tokens: int = 0
  color_aug: bool = False
  flip_aug: bool = False
  face_crop_aug_range: Optional[Tuple[float, float]] = None
  random_crop: bool = False
  caption_dropout_rate: float = 0.0
  caption_dropout_every_n_epochs: int = 0
  caption_tag_dropout_rate: float = 0.0
  token_warmup_min: int = 1
  token_warmup_step: float = 0

@dataclass
class DreamBoothSubsetParams(BaseSubsetParams):
  is_reg: bool = False
  class_tokens: Optional[str] = None
  caption_extension: str = ".caption"

@dataclass
class FineTuningSubsetParams(BaseSubsetParams):
  metadata_file: Optional[str] = None

@dataclass
class BaseDatasetParams:
  tokenizer: CLIPTokenizer = None
  max_token_length: int = None
  resolution: Optional[Tuple[int, int]] = None
  debug_dataset: bool = False

@dataclass
class DreamBoothDatasetParams(BaseDatasetParams):
  batch_size: int = 1
  enable_bucket: bool = False
  min_bucket_reso: int = 256
  max_bucket_reso: int = 1024
  bucket_reso_steps: int = 64
  bucket_no_upscale: bool = False
  prior_loss_weight: float = 1.0

@dataclass
class FineTuningDatasetParams(BaseDatasetParams):
  batch_size: int = 1
  enable_bucket: bool = False
  min_bucket_reso: int = 256
  max_bucket_reso: int = 1024
  bucket_reso_steps: int = 64
  bucket_no_upscale: bool = False

@dataclass
class SubsetBlueprint:
  params: Union[DreamBoothSubsetParams, FineTuningSubsetParams]

@dataclass
class DatasetBlueprint:
  is_dreambooth: bool
  params: Union[DreamBoothDatasetParams, FineTuningDatasetParams]
  subsets: Sequence[SubsetBlueprint]

@dataclass
class DatasetGroupBlueprint:
  datasets: Sequence[DatasetBlueprint]
@dataclass
class Blueprint:
  dataset_group: DatasetGroupBlueprint


class ConfigSanitizer:
  # @curry
  @staticmethod
  def __validate_and_convert_twodim(klass, value: Sequence) -> Tuple:
    Schema(ExactSequence([klass, klass]))(value)
    return tuple(value)

  # @curry
  @staticmethod
  def __validate_and_convert_scalar_or_twodim(klass, value: Union[float, Sequence]) -> Tuple:
    Schema(Any(klass, ExactSequence([klass, klass])))(value)
    try:
      Schema(klass)(value)
      return (value, value)
    except:
      return ConfigSanitizer.__validate_and_convert_twodim(klass, value)

  # subset schema
  SUBSET_ASCENDABLE_SCHEMA = {
    "color_aug": bool,
    "face_crop_aug_range": functools.partial(__validate_and_convert_twodim.__func__, float),
    "flip_aug": bool,
    "num_repeats": int,
    "random_crop": bool,
    "shuffle_caption": bool,
    "keep_tokens": int,
    "token_warmup_min": int,
    "token_warmup_step": Any(float,int),
  }
  # DO means DropOut
  DO_SUBSET_ASCENDABLE_SCHEMA = {
    "caption_dropout_every_n_epochs": int,
    "caption_dropout_rate": Any(float, int),
    "caption_tag_dropout_rate": Any(float, int),
  }
  # DB means DreamBooth
  DB_SUBSET_ASCENDABLE_SCHEMA = {
    "caption_extension": str,
    "class_tokens": str,
  }
  DB_SUBSET_DISTINCT_SCHEMA = {
    Required("image_dir"): str,
    "is_reg": bool,
  }
  # FT means FineTuning
  FT_SUBSET_DISTINCT_SCHEMA = {
    Required("metadata_file"): str,
    "image_dir": str,
  }

  # datasets schema
  DATASET_ASCENDABLE_SCHEMA = {
    "batch_size": int,
    "bucket_no_upscale": bool,
    "bucket_reso_steps": int,
    "enable_bucket": bool,
    "max_bucket_reso": int,
    "min_bucket_reso": int,
    "resolution": functools.partial(__validate_and_convert_scalar_or_twodim.__func__, int),
  }

  # options handled by argparse but not handled by user config
  ARGPARSE_SPECIFIC_SCHEMA = {
    "debug_dataset": bool,
    "max_token_length": Any(None, int),
    "prior_loss_weight": Any(float, int),
  }
  # for handling default None value of argparse
  ARGPARSE_NULLABLE_OPTNAMES = [
    "face_crop_aug_range",
    "resolution",
  ]
  # prepare map because option name may differ among argparse and user config
  ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME = {
    "train_batch_size": "batch_size",
    "dataset_repeats": "num_repeats",
  }

  def __init__(self, support_dreambooth: bool, support_finetuning: bool, support_dropout: bool) -> None:
    assert support_dreambooth or support_finetuning, "Neither DreamBooth mode nor fine tuning mode specified. Please specify one mode or more. / DreamBooth モードか fine tuning モードのどちらも指定されていません。1つ以上指定してください。"

    self.db_subset_schema = self.__merge_dict(
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.DB_SUBSET_DISTINCT_SCHEMA,
      self.DB_SUBSET_ASCENDABLE_SCHEMA,
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
    )

    self.ft_subset_schema = self.__merge_dict(
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.FT_SUBSET_DISTINCT_SCHEMA,
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
    )

    self.db_dataset_schema = self.__merge_dict(
      self.DATASET_ASCENDABLE_SCHEMA,
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.DB_SUBSET_ASCENDABLE_SCHEMA,
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
      {"subsets": [self.db_subset_schema]},
    )

    self.ft_dataset_schema = self.__merge_dict(
      self.DATASET_ASCENDABLE_SCHEMA,
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
      {"subsets": [self.ft_subset_schema]},
    )

    if support_dreambooth and support_finetuning:
      def validate_flex_dataset(dataset_config: dict):
        subsets_config = dataset_config.get("subsets", [])

        # check dataset meets FT style
        # NOTE: all FT subsets should have "metadata_file"
        if all(["metadata_file" in subset for subset in subsets_config]):
          return Schema(self.ft_dataset_schema)(dataset_config)
        # check dataset meets DB style
        # NOTE: all DB subsets should have no "metadata_file"
        elif all(["metadata_file" not in subset for subset in subsets_config]):
          return Schema(self.db_dataset_schema)(dataset_config)
        else:
          raise voluptuous.Invalid("DreamBooth subset and fine tuning subset cannot be mixed in the same dataset. Please split them into separate datasets. / DreamBoothのサブセットとfine tuninのサブセットを同一のデータセットに混在させることはできません。別々のデータセットに分割してください。")

      self.dataset_schema = validate_flex_dataset
    elif support_dreambooth:
      self.dataset_schema = self.db_dataset_schema
    else:
      self.dataset_schema = self.ft_dataset_schema

    self.general_schema = self.__merge_dict(
      self.DATASET_ASCENDABLE_SCHEMA,
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.DB_SUBSET_ASCENDABLE_SCHEMA if support_dreambooth else {},
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
    )

    self.user_config_validator = Schema({
      "general": self.general_schema,
      "datasets": [self.dataset_schema],
    })

    self.argparse_schema = self.__merge_dict(
      self.general_schema,
      self.ARGPARSE_SPECIFIC_SCHEMA,
      {optname: Any(None, self.general_schema[optname]) for optname in self.ARGPARSE_NULLABLE_OPTNAMES},
      {a_name: self.general_schema[c_name] for a_name, c_name in self.ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME.items()},
    )

    self.argparse_config_validator = Schema(Object(self.argparse_schema), extra=voluptuous.ALLOW_EXTRA)

  def sanitize_user_config(self, user_config: dict) -> dict:
    try:
      return self.user_config_validator(user_config)
    except MultipleInvalid:
      # TODO: エラー発生時のメッセージをわかりやすくする
      print("Invalid user config / ユーザ設定の形式が正しくないようです")
      raise

  # NOTE: In nature, argument parser result is not needed to be sanitize
  #   However this will help us to detect program bug
  def sanitize_argparse_namespace(self, argparse_namespace: argparse.Namespace) -> argparse.Namespace:
    try:
      return self.argparse_config_validator(argparse_namespace)
    except MultipleInvalid:
      # XXX: this should be a bug
      print("Invalid cmdline parsed arguments. This should be a bug. / コマンドラインのパース結果が正しくないようです。プログラムのバグの可能性が高いです。")
      raise

  # NOTE: value would be overwritten by latter dict if there is already the same key
  @staticmethod
  def __merge_dict(*dict_list: dict) -> dict:
    merged = {}
    for schema in dict_list:
      # merged |= schema
      for k, v in schema.items():
        merged[k] = v
    return merged


class BlueprintGenerator:
  BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = {
  }

  def __init__(self, sanitizer: ConfigSanitizer):
    self.sanitizer = sanitizer

  # runtime_params is for parameters which is only configurable on runtime, such as tokenizer
  def generate(self, user_config: dict, argparse_namespace: argparse.Namespace, **runtime_params) -> Blueprint:
    sanitized_user_config = self.sanitizer.sanitize_user_config(user_config)
    sanitized_argparse_namespace = self.sanitizer.sanitize_argparse_namespace(argparse_namespace)

    # convert argparse namespace to dict like config
    # NOTE: it is ok to have extra entries in dict
    optname_map = self.sanitizer.ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME
    argparse_config = {optname_map.get(optname, optname): value for optname, value in vars(sanitized_argparse_namespace).items()}

    general_config = sanitized_user_config.get("general", {})

    dataset_blueprints = []
    for dataset_config in sanitized_user_config.get("datasets", []):
      # NOTE: if subsets have no "metadata_file", these are DreamBooth datasets/subsets
      subsets = dataset_config.get("subsets", [])
      is_dreambooth = all(["metadata_file" not in subset for subset in subsets])
      if is_dreambooth:
        subset_params_klass = DreamBoothSubsetParams
        dataset_params_klass = DreamBoothDatasetParams
      else:
        subset_params_klass = FineTuningSubsetParams
        dataset_params_klass = FineTuningDatasetParams

      subset_blueprints = []
      for subset_config in subsets:
        params = self.generate_params_by_fallbacks(subset_params_klass,
                                                   [subset_config, dataset_config, general_config, argparse_config, runtime_params])
        subset_blueprints.append(SubsetBlueprint(params))

      params = self.generate_params_by_fallbacks(dataset_params_klass,
                                                 [dataset_config, general_config, argparse_config, runtime_params])
      dataset_blueprints.append(DatasetBlueprint(is_dreambooth, params, subset_blueprints))

    dataset_group_blueprint = DatasetGroupBlueprint(dataset_blueprints)

    return Blueprint(dataset_group_blueprint)

  @staticmethod
  def generate_params_by_fallbacks(param_klass, fallbacks: Sequence[dict]):
    name_map = BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME
    search_value = BlueprintGenerator.search_value
    default_params = asdict(param_klass())
    param_names = default_params.keys()

    params = {name: search_value(name_map.get(name, name), fallbacks, default_params.get(name)) for name in param_names}

    return param_klass(**params)

  @staticmethod
  def search_value(key: str, fallbacks: Sequence[dict], default_value = None):
    for cand in fallbacks:
      value = cand.get(key)
      if value is not None:
        return value

    return default_value


def generate_dataset_group_by_blueprint(dataset_group_blueprint: DatasetGroupBlueprint):
  datasets: List[Union[DreamBoothDataset, FineTuningDataset]] = []

  for dataset_blueprint in dataset_group_blueprint.datasets:
    if dataset_blueprint.is_dreambooth:
      subset_klass = DreamBoothSubset
      dataset_klass = DreamBoothDataset
    else:
      subset_klass = FineTuningSubset
      dataset_klass = FineTuningDataset

    subsets = [subset_klass(**asdict(subset_blueprint.params)) for subset_blueprint in dataset_blueprint.subsets]
    dataset = dataset_klass(subsets=subsets, **asdict(dataset_blueprint.params))
    datasets.append(dataset)

  # print info
  info = ""
  for i, dataset in enumerate(datasets):
    is_dreambooth = isinstance(dataset, DreamBoothDataset)
    info += dedent(f"""\
      [Dataset {i}]
        batch_size: {dataset.batch_size}
        resolution: {(dataset.width, dataset.height)}
        enable_bucket: {dataset.enable_bucket}
    """)

    if dataset.enable_bucket:
      info += indent(dedent(f"""\
        min_bucket_reso: {dataset.min_bucket_reso}
        max_bucket_reso: {dataset.max_bucket_reso}
        bucket_reso_steps: {dataset.bucket_reso_steps}
        bucket_no_upscale: {dataset.bucket_no_upscale}
      \n"""), "  ")
    else:
      info += "\n"

    for j, subset in enumerate(dataset.subsets):
      info += indent(dedent(f"""\
        [Subset {j} of Dataset {i}]
          image_dir: "{subset.image_dir}"
          image_count: {subset.img_count}
          num_repeats: {subset.num_repeats}
          shuffle_caption: {subset.shuffle_caption}
          keep_tokens: {subset.keep_tokens}
          caption_dropout_rate: {subset.caption_dropout_rate}
          caption_dropout_every_n_epoches: {subset.caption_dropout_every_n_epochs}
          caption_tag_dropout_rate: {subset.caption_tag_dropout_rate}
          color_aug: {subset.color_aug}
          flip_aug: {subset.flip_aug}
          face_crop_aug_range: {subset.face_crop_aug_range}
          random_crop: {subset.random_crop}
          token_warmup_min: {subset.token_warmup_min},
          token_warmup_step: {subset.token_warmup_step},
      """), "  ")

      if is_dreambooth:
        info += indent(dedent(f"""\
          is_reg: {subset.is_reg}
          class_tokens: {subset.class_tokens}
          caption_extension: {subset.caption_extension}
        \n"""), "    ")
      else:
        info += indent(dedent(f"""\
          metadata_file: {subset.metadata_file}
        \n"""), "    ")

  print(info)

  # make buckets first because it determines the length of dataset
  # and set the same seed for all datasets
  seed = random.randint(0, 2**31) # actual seed is seed + epoch_no
  for i, dataset in enumerate(datasets):
    print(f"[Dataset {i}]")
    dataset.make_buckets()
    dataset.set_seed(seed)

  return DatasetGroup(datasets)


def generate_dreambooth_subsets_config_by_subdirs(train_data_dir: Optional[str] = None, reg_data_dir: Optional[str] = None):
  def extract_dreambooth_params(name: str) -> Tuple[int, str]:
    tokens = name.split('_')
    try:
      n_repeats = int(tokens[0])
    except ValueError as e:
      print(f"ignore directory without repeats / 繰り返し回数のないディレクトリを無視します: {name}")
      return 0, ""
    caption_by_folder = '_'.join(tokens[1:])
    return n_repeats, caption_by_folder

  def generate(base_dir: Optional[str], is_reg: bool):
    if base_dir is None:
      return []

    base_dir: Path = Path(base_dir)
    if not base_dir.is_dir():
      return []

    subsets_config = []
    for subdir in base_dir.iterdir():
      if not subdir.is_dir():
        continue

      num_repeats, class_tokens = extract_dreambooth_params(subdir.name)
      if num_repeats < 1:
        continue

      subset_config = {"image_dir": str(subdir), "num_repeats": num_repeats, "is_reg": is_reg, "class_tokens": class_tokens}
      subsets_config.append(subset_config)

    return subsets_config

  subsets_config = []
  subsets_config += generate(train_data_dir, False)
  subsets_config += generate(reg_data_dir, True)

  return subsets_config


def load_user_config(file: str) -> dict:
  file: Path = Path(file)
  if not file.is_file():
    raise ValueError(f"file not found / ファイルが見つかりません: {file}")

  if file.name.lower().endswith('.json'):
    try:
      with open(file, 'r') as f:
        config = json.load(f)
    except Exception:
      print(f"Error on parsing JSON config file. Please check the format. / JSON 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}")
      raise
  elif file.name.lower().endswith('.toml'):
    try:
      config = toml.load(file)
    except Exception:
      print(f"Error on parsing TOML config file. Please check the format. / TOML 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}")
      raise
  else:
    raise ValueError(f"not supported config file format / 対応していない設定ファイルの形式です: {file}")

  return config

# for config test
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--support_dreambooth", action="store_true")
  parser.add_argument("--support_finetuning", action="store_true")
  parser.add_argument("--support_dropout", action="store_true")
  parser.add_argument("dataset_config")
  config_args, remain = parser.parse_known_args()

  parser = argparse.ArgumentParser()
  train_util.add_dataset_arguments(parser, config_args.support_dreambooth, config_args.support_finetuning, config_args.support_dropout)
  train_util.add_training_arguments(parser, config_args.support_dreambooth)
  argparse_namespace = parser.parse_args(remain)
  train_util.prepare_dataset_args(argparse_namespace, config_args.support_finetuning)

  print("[argparse_namespace]")
  print(vars(argparse_namespace))

  user_config = load_user_config(config_args.dataset_config)

  print("\n[user_config]")
  print(user_config)

  sanitizer = ConfigSanitizer(config_args.support_dreambooth, config_args.support_finetuning, config_args.support_dropout)
  sanitized_user_config = sanitizer.sanitize_user_config(user_config)

  print("\n[sanitized_user_config]")
  print(sanitized_user_config)

  blueprint = BlueprintGenerator(sanitizer).generate(user_config, argparse_namespace)

  print("\n[blueprint]")
  print(blueprint)
