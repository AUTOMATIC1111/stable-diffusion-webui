from torch.nn.parallel import DistributedDataParallel as DDP
import importlib
import argparse
import gc
import math
import os
import random
import time
import json
# import toml
from multiprocessing import Value

from tqdm import tqdm
import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler

import sd_scripts.library.train_util as train_util
from sd_scripts.library.train_util import (
    DreamBoothDataset,
)
import sd_scripts.library.config_util as config_util
from sd_scripts.library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import sd_scripts.library.huggingface_util as huggingface_util
import sd_scripts.library.custom_train_functions as custom_train_functions
from sd_scripts.library.custom_train_functions import apply_snr_weight, get_weighted_text_embeddings, \
    pyramid_noise_like, \
    apply_noise_offset


# TODO 他のスクリプトと共通化する
def generate_step_logs(args: argparse.Namespace, current_loss, avr_loss, lr_scheduler):
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    lrs = lr_scheduler.get_last_lr()

    if args.network_train_text_encoder_only or len(lrs) <= 2:  # not block lr (or single block)
        if args.network_train_unet_only:
            logs["lr/unet"] = float(lrs[0])
        elif args.network_train_text_encoder_only:
            logs["lr/textencoder"] = float(lrs[0])
        else:
            logs["lr/textencoder"] = float(lrs[0])
            logs["lr/unet"] = float(lrs[-1])  # may be same to textencoder

        if args.optimizer_type.lower().startswith("DAdapt".lower()):  # tracking d*lr value of unet.
            logs["lr/d*lr"] = lr_scheduler.optimizers[-1].param_groups[0]["d"] * \
                              lr_scheduler.optimizers[-1].param_groups[0]["lr"]
    else:
        idx = 0
        if not args.network_train_unet_only:
            logs["lr/textencoder"] = float(lrs[0])
            idx = 1

        for i in range(idx, len(lrs)):
            logs[f"lr/group{i}"] = float(lrs[i])
            if args.optimizer_type.lower().startswith("DAdapt".lower()):
                logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i][
                    "lr"]
                )

    return logs


def train(args, train_epoch_callback=None):
    session_id = random.randint(0, 2 ** 32)
    training_started_at = time.time()
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None
    use_user_config = args.dataset_config is not None

    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    tokenizer = train_util.load_tokenizer(args)

    # データセットを準備する
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, True))
    if use_user_config:
        print(f"Load dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        ignored = ["train_data_dir", "reg_data_dir", "in_json"]
        if any(getattr(args, attr) is not None for attr in ignored):
            print(
                "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                    ", ".join(ignored)
                )
            )
    else:
        if use_dreambooth_method:
            print("Use DreamBooth method.")
            list_repeats = args.repeats_times
            class_tokens = args.trigger_words
            list_reg_repeats = []
            reg_tokens = []
            list_train_data_dirs = args.list_train_data_dir
            list_reg_data_dirs = []
            if args.reg_data_dir is not None:
                list_reg_data_dirs = args.list_reg_data_dir
                reg_tokens = args.reg_tokens
                list_reg_repeats = args.list_reg_repeats

            # print("111",list_repeats,class_tokens,reg_tokens)
            user_config = {
                "datasets": [
                    # {"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)}
                    {"subsets": config_util.generate_dreambooth_subsets_config_by_args(
                        list_repeats, class_tokens, list_train_data_dirs, list_reg_repeats, reg_tokens,
                        list_reg_data_dirs)}
                ]
            }
        else:
            print("Train with captions.")
            user_config = {
                "datasets": [
                    {
                        "subsets": [
                            {
                                "image_dir": args.train_data_dir,
                                "metadata_file": args.in_json,
                            }
                        ]
                    }
                ]
            }

    blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
    train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group)
        return
    if len(train_dataset_group) == 0:
        print(
            "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    # acceleratorを準備する
    print("prepare accelerator")

    accelerator, unwrap_model = train_util.prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # モデルを読み込む
    text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

    # モデルに xformers とか memory efficient attention を組み込む
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk,
                                              accelerator.is_main_process)
        vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        accelerator.wait_for_everyone()

    # prepare network
    import sys

    sys.path.append(os.path.dirname(__file__))
    print("import network module:", args.network_module)
    network_module = importlib.import_module(args.network_module)

    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value

    # if a new network is added in future, add if ~ then blocks for each network (;'∀')
    network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet,
                                            **net_kwargs)
    if network is None:
        return

    if hasattr(network, "prepare_network"):
        network.prepare_network(args)

    train_unet = not args.network_train_text_encoder_only
    train_text_encoder = not args.network_train_unet_only
    network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        print(f"load network weights from {args.network_weights}: {info}")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
        network.enable_gradient_checkpointing()  # may have no effect

    # 学習に必要なクラスを準備する
    print("prepare optimizer, data loader etc.")

    # 後方互換性を確保するよ
    try:
        trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    except TypeError:
        print(
            "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)"
        )
        trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

    optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

    # dataloaderを準備する
    # DataLoaderのプロセス数：0はメインプロセスになる
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collater,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        if is_main_process:
            print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
    if args.full_fp16:
        assert (
                args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        print("enable full fp16 training.")
        network.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    if train_unet and train_text_encoder:
        unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler
        )
    elif train_unet:
        unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, network, optimizer, train_dataloader, lr_scheduler
        )
    elif train_text_encoder:
        text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, network, optimizer, train_dataloader, lr_scheduler
        )
    else:
        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader,
                                                                                 lr_scheduler)

    # transform DDP after prepare (train_network here only)
    text_encoder, unet, network = train_util.transform_if_model_is_DDP(text_encoder, unet, network)

    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)
    if args.gradient_checkpointing:  # according to TI example in Diffusers, train is required
        unet.train()
        text_encoder.train()

        # set top parameter requires_grad = True for gradient checkpointing works
        text_encoder.text_model.embeddings.requires_grad_(True)
    else:
        unet.eval()
        text_encoder.eval()

    network.prepare_grad_etc(text_encoder, unet)

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    # TODO: find a way to handle total batch size when there are multiple datasets
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if is_main_process:
        print("running training / 学習開始")
        print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        print(f"  num epochs / epoch数: {num_train_epochs}")
        print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
        # print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    # TODO refactor metadata creation and move to util
    metadata = {
        "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
        "ss_training_started_at": training_started_at,  # unix timestamp
        "ss_output_name": args.output_name,
        "ss_learning_rate": args.learning_rate,
        "ss_text_encoder_lr": args.text_encoder_lr,
        "ss_unet_lr": args.unet_lr,
        "ss_num_train_images": train_dataset_group.num_train_images,
        "ss_num_reg_images": train_dataset_group.num_reg_images,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": num_train_epochs,
        "ss_gradient_checkpointing": args.gradient_checkpointing,
        "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
        "ss_max_train_steps": args.max_train_steps,
        "ss_lr_warmup_steps": args.lr_warmup_steps,
        "ss_lr_scheduler": args.lr_scheduler,
        "ss_network_module": args.network_module,
        "ss_network_dim": args.network_dim,
        # None means default because another network than LoRA may have another default dim
        "ss_network_alpha": args.network_alpha,  # some networks may not use this value
        "ss_mixed_precision": args.mixed_precision,
        "ss_full_fp16": bool(args.full_fp16),
        "ss_v2": bool(args.v2),
        "ss_clip_skip": args.clip_skip,
        "ss_max_token_length": args.max_token_length,
        "ss_cache_latents": bool(args.cache_latents),
        "ss_seed": args.seed,
        "ss_lowram": args.lowram,
        "ss_noise_offset": args.noise_offset,
        "ss_multires_noise_iterations": args.multires_noise_iterations,
        "ss_multires_noise_discount": args.multires_noise_discount,
        "ss_training_comment": args.training_comment,  # will not be updated after training
        "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": args.max_grad_norm,
        "ss_caption_dropout_rate": args.caption_dropout_rate,
        "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
        "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
        "ss_face_crop_aug_range": args.face_crop_aug_range,
        "ss_prior_loss_weight": args.prior_loss_weight,
        "ss_min_snr_gamma": args.min_snr_gamma,
    }

    if use_user_config:
        # save metadata of multiple datasets
        # NOTE: pack "ss_datasets" value as json one time
        #   or should also pack nested collections as json?
        datasets_metadata = []
        tag_frequency = {}  # merge tag frequency for metadata editor
        dataset_dirs_info = {}  # merge subset dirs for metadata editor

        for dataset in train_dataset_group.datasets:
            is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
            dataset_metadata = {
                "is_dreambooth": is_dreambooth_dataset,
                "batch_size_per_device": dataset.batch_size,
                "num_train_images": dataset.num_train_images,  # includes repeating
                "num_reg_images": dataset.num_reg_images,
                "resolution": (dataset.width, dataset.height),
                "enable_bucket": bool(dataset.enable_bucket),
                "min_bucket_reso": dataset.min_bucket_reso,
                "max_bucket_reso": dataset.max_bucket_reso,
                "tag_frequency": dataset.tag_frequency,
                "bucket_info": dataset.bucket_info,
            }

            subsets_metadata = []
            for subset in dataset.subsets:
                subset_metadata = {
                    "img_count": subset.img_count,
                    "num_repeats": subset.num_repeats,
                    "color_aug": bool(subset.color_aug),
                    "flip_aug": bool(subset.flip_aug),
                    "random_crop": bool(subset.random_crop),
                    "shuffle_caption": bool(subset.shuffle_caption),
                    "keep_tokens": subset.keep_tokens,
                }

                image_dir_or_metadata_file = None
                if subset.image_dir:
                    image_dir = os.path.basename(subset.image_dir)
                    subset_metadata["image_dir"] = image_dir
                    image_dir_or_metadata_file = image_dir

                if is_dreambooth_dataset:
                    subset_metadata["class_tokens"] = subset.class_tokens
                    subset_metadata["is_reg"] = subset.is_reg
                    if subset.is_reg:
                        image_dir_or_metadata_file = None  # not merging reg dataset
                else:
                    metadata_file = os.path.basename(subset.metadata_file)
                    subset_metadata["metadata_file"] = metadata_file
                    image_dir_or_metadata_file = metadata_file  # may overwrite

                subsets_metadata.append(subset_metadata)

                # merge dataset dir: not reg subset only
                # TODO update additional-network extension to show detailed dataset config from metadata
                if image_dir_or_metadata_file is not None:
                    # datasets may have a certain dir multiple times
                    v = image_dir_or_metadata_file
                    i = 2
                    while v in dataset_dirs_info:
                        v = image_dir_or_metadata_file + f" ({i})"
                        i += 1
                    image_dir_or_metadata_file = v

                    dataset_dirs_info[image_dir_or_metadata_file] = {"n_repeats": subset.num_repeats,
                                                                     "img_count": subset.img_count}

            dataset_metadata["subsets"] = subsets_metadata
            datasets_metadata.append(dataset_metadata)

            # merge tag frequency:
            for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                # なので、ここで複数datasetの回数を合算してもあまり意味はない
                if ds_dir_name in tag_frequency:
                    continue
                tag_frequency[ds_dir_name] = ds_freq_for_dir

        metadata["ss_datasets"] = json.dumps(datasets_metadata)
        metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
        metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
    else:
        # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
        assert (
                len(train_dataset_group.datasets) == 1
        ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

        dataset = train_dataset_group.datasets[0]

        dataset_dirs_info = {}
        reg_dataset_dirs_info = {}
        if use_dreambooth_method:
            for subset in dataset.subsets:
                info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats,
                                                            "img_count": subset.img_count}
        else:
            for subset in dataset.subsets:
                dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                    "n_repeats": subset.num_repeats,
                    "img_count": subset.img_count,
                }

        metadata.update(
            {
                "ss_batch_size_per_device": args.train_batch_size,
                "ss_total_batch_size": total_batch_size,
                "ss_resolution": args.resolution,
                "ss_color_aug": bool(args.color_aug),
                "ss_flip_aug": bool(args.flip_aug),
                "ss_random_crop": bool(args.random_crop),
                "ss_shuffle_caption": bool(args.shuffle_caption),
                "ss_enable_bucket": bool(dataset.enable_bucket),
                "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                "ss_min_bucket_reso": dataset.min_bucket_reso,
                "ss_max_bucket_reso": dataset.max_bucket_reso,
                "ss_keep_tokens": args.keep_tokens,
                "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                "ss_bucket_info": json.dumps(dataset.bucket_info),
            }
        )

    # add extra args
    if args.network_args:
        metadata["ss_network_args"] = json.dumps(net_kwargs)

    # model name and hash
    if args.pretrained_model_name_or_path is not None:
        sd_model_name = args.pretrained_model_name_or_path
        if os.path.exists(sd_model_name):
            metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
            metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if args.vae is not None:
        vae_name = args.vae
        if os.path.exists(vae_name):
            metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
            metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name

    metadata = {k: str(v) for k, v in metadata.items()}

    # make minimum metadata for filtering
    minimum_keys = ["ss_network_module", "ss_network_dim", "ss_network_alpha", "ss_network_args"]
    minimum_metadata = {}
    for key in minimum_keys:
        if key in metadata:
            minimum_metadata[key] = metadata[key]

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process,
                        desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("network_train" if args.log_tracker_name is None else args.log_tracker_name)

    loss_list = []
    loss_total = 0.0
    del train_dataset_group

    # if hasattr(network, "on_step_start"):
    #     on_step_start = network.on_step_start
    # else:
    #     on_step_start = lambda *args, **kwargs: None

    # function for saving/removing
    def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        print(f"saving checkpoint: {ckpt_file}")
        metadata["ss_training_finished_at"] = str(time.time())
        metadata["ss_steps"] = str(steps)
        metadata["ss_epoch"] = str(epoch_no)

        unwrapped_nw.save_weights(ckpt_file, save_dtype, minimum_metadata if args.no_metadata else metadata)
        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def remove_model(old_ckpt_name):
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

    # training loop
    for epoch in range(num_train_epochs):
        if is_main_process:
            print(f"epoch {epoch + 1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        metadata["ss_epoch"] = str(epoch + 1)

        network.on_epoch_start(text_encoder, unet)

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(network):
                # on_step_start(text_encoder, unet)

                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device)
                    else:
                        # latentに変換
                        latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215
                b_size = latents.shape[0]

                with torch.set_grad_enabled(train_text_encoder):
                    # Get the text embedding for conditioning
                    if args.weighted_captions:
                        encoder_hidden_states = get_weighted_text_embeddings(
                            tokenizer,
                            text_encoder,
                            batch["captions"],
                            accelerator.device,
                            args.max_token_length // 75 if args.max_token_length else 1,
                            clip_skip=args.clip_skip,
                        )
                    else:
                        input_ids = batch["input_ids"].to(accelerator.device)
                        encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizer, text_encoder,
                                                                             weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents, device=latents.device)
                if args.noise_offset:
                    noise = apply_noise_offset(latents, noise, args.noise_offset, args.adaptive_noise_scale)
                elif args.multires_noise_iterations:
                    noise = pyramid_noise_like(noise, latents.device, args.multires_noise_iterations,
                                               args.multires_noise_discount)

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,),
                                          device=latents.device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict the noise residual
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.v_parameterization:
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])

                loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                loss = loss * loss_weights

                if args.min_snr_gamma:
                    loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)

                loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = network.get_trainable_params()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                train_util.sample_images(
                    accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet
                )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                        save_model(ckpt_name, unwrap_model(network), global_step, epoch)

                        if args.save_state:
                            train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                        remove_step_no = train_util.get_remove_step_no(args, global_step)
                        if remove_step_no is not None:
                            remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as,
                                                                             remove_step_no)
                            remove_model(remove_ckpt_name)

            current_loss = loss.detach().item()
            if epoch == 0:
                loss_list.append(current_loss)
            else:
                loss_total -= loss_list[step]
                loss_list[step] = current_loss
            loss_total += current_loss
            avr_loss = loss_total / len(loss_list)
            logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if args.logging_dir is not None:
                logs = generate_step_logs(args, current_loss, avr_loss, lr_scheduler)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_total / len(loss_list)}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        # 指定エポックごとにモデルを保存
        if args.save_every_n_epochs is not None:
            saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
            if is_main_process and saving:
                ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                save_model(ckpt_name, unwrap_model(network), global_step, epoch + 1)

                remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                if remove_epoch_no is not None:
                    remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                    remove_model(remove_ckpt_name)

                if args.save_state:
                    train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

        train_util.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer,
                                 text_encoder, unet)
        if callable(train_epoch_callback):
            train_epoch_callback(epoch + 1, loss_total / len(loss_list), num_train_epochs)
        # end of epoch
        if math.isnan(loss_total / len(loss_list)):
            # nan(task failed)
            accelerator.end_training()
            del accelerator
            return False

    # metadata["ss_epoch"] = str(num_train_epochs)
    metadata["ss_training_finished_at"] = str(time.time())

    if is_main_process:
        network = unwrap_model(network)

    accelerator.end_training()

    if is_main_process and args.save_state:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
        save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

        print("model saved.")
        return True


def setup_parser() -> argparse.ArgumentParser:
    # parser = argparse.ArgumentParser()
    parent_parser = argparse.ArgumentParser()
    subparsers = parent_parser.add_subparsers(title="sys_sub_parsers")
    parser = subparsers.add_parser("lora_train",
                                   help="create the lora_train environment")

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument("--no_metadata", action="store_true",
                        help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")

    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument(
        "--network_dim", type=int, default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）"
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)）",
    )
    parser.add_argument(
        "--network_args", type=str, default=None, nargs="*", help="additional argmuments for network (key=value)"
    )
    parser.add_argument("--network_train_unet_only", action="store_true", help="only training U-Net part")
    parser.add_argument(
        "--network_train_text_encoder_only", action="store_true", help="only training Text Encoder part"
    )
    parser.add_argument(
        "--training_comment", type=str, default=None, help="arbitrary comment string stored in metadata"
    )
    parser.add_argument(
        "--repeats_times", type=list, default=[""],
        help="repeat times of source images for traing"
    )
    parser.add_argument(
        "--trigger_words", type=list, default=[""],
        help="unique token(trigger word) for lora"
    )
    parser.add_argument(
        "--reg_tokens", type=list, default=[""],
        help="lora class type,for example, a ly2 dog,ly2 is class_token and dog is reg_token"
    )
    parser.add_argument(
        "--list_train_data_dir", type=list, default=[""],
        help="list for train_data folder name"
    )
    parser.add_argument(
        "--list_reg_data_dir", type=list, default=[""],
        help="list for reg_data folder name"
    )
    parser.add_argument(
        "--list_reg_repeats", type=list, default=[""],
        help="repeat times of source images for reg images"
    )

    return parser


def train_callback(epoch, avg_loss):
    print(epoch, avg_loss)


def train_with_file(config_file_path, callback):
    parser = setup_parser()
    args = parser.parse_args([])
    args.config_file = config_file_path
    if config_file_path.endswith(".json"):
        args = train_util.read_config_from_json(args, parser)
    elif config_file_path.endswith(".toml"):
        args = train_util.read_config_from_file(args, parser)
    train(args, callback)


# 训练函数接口
def train_with_params(
        pretrained_model_name_or_path,
        network_weights,
        output_name,
        save_model_as="safetensors",
        v2=False,
        v_parameterization=False,
        output_dir="./output",
        logging_dir="./logs",
        save_every_n_epochs=2,
        save_last_n_epochs=10,
        save_precision=None,

        trigger_words=None,
        max_token_length=75,  # max token length of text encoder (default for 75, 150 or 225)
        reg_tokens=None,
        list_train_data_dir=None,
        list_reg_data_dir=None,
        num_repeats=None,
        list_reg_repeats=None,
        batch_size=1,
        resolution="512,512",  # 64的倍数
        cache_latents=False,
        # cache latents to main memory to reduce VRAM usage (augmentations must be disabled)
        cache_latents_to_disk=False,
        # cache latents to disk to reduce VRAM usage (augmentations must be disabled)
        enable_bucket=True,  # enable buckets for multi aspect ratio training
        min_bucket_reso=256,  # 范围自己定，minimum resolution for buckets
        max_bucket_reso=1024,  # 范围自己定，maximum resolution for buckets
        bucket_reso_steps=64,  # 秋叶版没有这个,steps of resolution for buckets, divisible by 8 is recommended
        bucket_no_upscale=False,  # 秋叶版没有这个,make bucket for each image without upscaling
        token_warmup_min=1,  # 秋叶版没有这个,start learning at N tags (token means comma separated strinfloatgs)
        token_warmup_step=0,
        # 秋叶版没有这个,tag length reaches maximum on N steps (or N*max_train_steps if N<1)

        caption_extension=".txt",
        caption_dropout_rate=0.0,  # Rate out dropout caption(0.0~1.0)
        caption_dropout_every_n_epochs=0,  # Dropout all captions every N epochs
        caption_tag_dropout_rate=0.0,  # Rate out dropout comma separated tokens(0.0~1.0)
        shuffle_caption=False,  # shuffle comma-separated caption
        weighted_captions=False,  # 使用带权重的 token，不推荐与 shuffle_caption 一同开启
        keep_tokens=0,
        # keep heading N tokens when shuffling caption tokens (token means comma separated strings)
        color_aug=False,  # 秋叶版没有这个,enable weak color augmentation
        flip_aug=False,  # 秋叶版没有这个,enable horizontal flip augmentation
        face_crop_aug_range=None,
        # 秋叶版没有这个,enable face-centered crop augmentation and its range (e.g. 2.0,4.0)
        random_crop=False,
        # 秋叶版没有这个,enable random crop (for style training in face-centered crop augmentation)

        lowram=True,
        # enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle)
        mem_eff_attn=False,  # use memory efficient attention for CrossAttention
        xformers=True,  # 如果mem_eff_attn为True则xformers设置无效
        vae=None,  # 秋叶版没有这个,path to checkpoint of vae to replace
        max_data_loader_n_workers=8,
        # 秋叶版没有这个,max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading)
        persistent_data_loader_workers=True,
        # persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory)

        max_train_steps=1600,  # 秋叶版没有这个,
        epoch=10,  # 整数，随便填
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        # 整数，随便填, Number of updates steps to accumulate before performing a backward/update pass
        mixed_precision="no",  # 是否使用混精度
        full_fp16=True,  # fp16 training including gradients

        enable_preview=False,  # 和下面这几个参数一起的
        sample_prompts=None,  # file for prompts to generate sample images
        sample_sampler="ddim",
        # ["ddim","pndm","lms","euler","euler_a","heun","dpm_2","dpm_2_a","dpmsolver","dpmsolver++","dpmsingle","k_lms","k_euler","k_euler_a","k_dpm_2","k_dpm_2_a",]
        sample_every_n_epochs=None,

        network_module="networks.lora",
        network_train_unet_only=False,
        network_train_text_encoder_only=False,
        network_dim=32,  # 4的倍数，<=256
        network_alpha=16,  # 小于等于network_dim,可以不是4的倍数
        clip_skip=1,  # 0-12

        # network额外参数
        conv_dim=None,
        # lycoris才有，# 4的倍数, 适用于lora,dylora。如果是dylora,则"conv_dim must be same as network_dim",
        conv_alpha=None,  # lycoris才有，<=conv_dim, 适用于lora,dylora
        unit=8,  # 秋叶版没有
        dropout=0,  # dropout 概率, 0 为不使用 dropout, 越大则 dropout 越多，推荐 0~0.5， LoHa/LoKr/(IA)^3暂时不支持
        algo='lora',  # 可选['lora','loha','lokr','ia3']

        enable_block_weights=False,  # 让下面几个参数有效
        block_dims=None,  # lora,
        block_alphas=None,  # lora,
        conv_block_dims=None,  # lora,
        conv_block_alphas=None,  # lora,
        down_lr_weight=None,  # lora
        mid_lr_weight=None,  # lora
        up_lr_weight=None,  # lora
        block_lr_zero_threshold=0.0,  # float型，分层学习率置 0 阈值

        optimizer_type="AdamW8bit",
        # AdamW (default), AdamW8bit, Lion8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation(DAdaptAdam), DAdaptAdaGrad, DAdaptAdan, DAdaptSGD, AdaFactor
        weight_decay=None,  # weight_decay=0.01 ,optimizer_args,优化器内部的参数，权重衰减系数，不建议随便改
        betas=None,  # betas=0.9,0.999,optimizer_args,优化器内部的参数，不建议随便改

        max_grad_norm=1.0,  # Max gradient norm, 0 for no clipping

        learning_rate=0.0001,
        unet_lr=0.0001,
        text_encoder_lr=0.00001,
        lr_scheduler="cosine_with_restarts",
        # linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor
        lr_scheduler_num_cycles=1,  # Number of restarts for cosine scheduler with restarts
        lr_warmup_steps=0,  # Number of steps for the warmup in the lr scheduler
        lr_scheduler_power=1,  # Polynomial power for polynomial scheduler

        seed=1,
        prior_loss_weight=1.0,  # loss weight for regularization images
        min_snr_gamma=None,  # ,float型，最小信噪比伽马值，如果启用推荐为 5
        noise_offset=None,  # enable noise offset with this value (if enabled, around 0.1 is recommended)
        adaptive_noise_scale=None,
        # 与noise_offset配套使用；add `latent mean absolute value * this value` to noise_offset (disabled if None, default)
        multires_noise_iterations=None,  # 整数，多分辨率（金字塔）噪声迭代次数 推荐 6-10。无法与 noise_offset 一同启用。
        multires_noise_discount=0.3,  # 多分辨率（金字塔）噪声迭代次数 推荐 6-10。无法与 noise_offset 一同启用。

        config_file=None,  # using .toml instead of args to pass hyperparameter
        output_config=False,  # output command line args to given .json file
        callback=None,
):
    # TODO 数据校验，或者流程重新梳理，去掉args
    parser = setup_parser()
    args = parser.parse_args([])
    # args = train_util.read_config_from_file(args, parser)

    args.pretrained_model_name_or_path = pretrained_model_name_or_path
    if network_weights:
        args.network_weights = network_weights

    if list_reg_data_dir:
        args.reg_tokens = reg_tokens or []
        args.list_reg_data_dir = list_reg_data_dir or []

    args.output_name = output_name if output_name != "" and output_name != -1 else None
    args.save_model_as = save_model_as
    args.trigger_words = trigger_words or []
    args.save_every_n_epochs = save_every_n_epochs
    args.list_train_data_dir = list_train_data_dir or []
    args.repeats_times = num_repeats or []
    args.list_reg_repeats = list_reg_repeats or []
    args.train_batch_size = batch_size
    args.max_train_epochs = epoch
    args.resolution = resolution
    args.clip_skip = clip_skip
    args.network_dim = network_dim
    args.network_alpha = network_alpha

    args.learning_rate = learning_rate
    args.unet_lr = unet_lr if unet_lr != "" and unet_lr != -1 else None
    args.text_encoder_lr = text_encoder_lr if text_encoder_lr != "" and text_encoder_lr != -1 else None
    args.optimizer_type = optimizer_type if optimizer_type != "" and optimizer_type != -1 else None
    args.lr_scheduler_num_cycles = lr_scheduler_num_cycles if lr_scheduler_num_cycles != "" and lr_scheduler_num_cycles != -1 else None
    args.lr_scheduler = lr_scheduler if lr_scheduler != "" and lr_scheduler != -1 else None

    args.network_train_unet_only = network_train_unet_only if network_train_unet_only != "" and network_train_unet_only != -1 else None
    args.network_train_text_encoder_only = network_train_text_encoder_only if network_train_text_encoder_only != "" and network_train_text_encoder_only != -1 else None
    args.seed = seed

    args.output_dir = output_dir if output_dir != "" and output_dir != -1 else None
    args.logging_dir = logging_dir if logging_dir != "" and logging_dir != -1 else None
    args.save_last_n_epochs = save_last_n_epochs if save_last_n_epochs != "" and save_last_n_epochs != -1 else None

    args.v2 = v2 if v2 != "" and v2 != -1 else None
    args.v_parameterization = v_parameterization if v_parameterization != "" and v_parameterization != -1 else None

    if enable_bucket:
        args.enable_bucket = enable_bucket if enable_bucket != "" and enable_bucket != -1 else None
        args.min_bucket_reso = min_bucket_reso if min_bucket_reso != "" and min_bucket_reso != -1 else None
        args.max_bucket_reso = max_bucket_reso if max_bucket_reso != "" and max_bucket_reso != -1 else None
        args.bucket_reso_steps = bucket_reso_steps if bucket_reso_steps != "" and bucket_reso_steps != -1 else None
        args.bucket_no_upscale = bucket_no_upscale if bucket_no_upscale != "" and bucket_no_upscale != -1 else None

    args.network_module = network_module
    if args.network_args is None:
        args.network_args = []
    if network_module == "lycoris.kohya":
        if conv_dim is not None and conv_dim != -1 and conv_dim != "": args.network_args.append(f"conv_dim={conv_dim}")
        if conv_alpha is not None and conv_alpha != -1 and conv_alpha != "": args.network_args.append(
            f"conv_alpha={conv_alpha}")
        if dropout is not None and dropout != -1 and dropout != "": args.network_args.append(f"dropout={dropout}")
        args.network_args.append(f"algo={algo}")
    elif network_module == "networks.dylora":
        args.network_args.append(f"unit={unit}")
    elif network_module == "networks.lora":
        if conv_dim is not None and conv_dim != -1 and conv_dim != "": args.network_args.append(f"conv_dim={conv_dim}")
        if conv_alpha is not None and conv_alpha != -1 and conv_alpha != "": args.network_args.append(
            f"conv_alpha={conv_alpha}")
        if dropout is not None and dropout != -1 and dropout != "": args.network_args.append(f"dropout={dropout}")
        if block_dims is not None and block_dims != -1 and block_dims != "": args.network_args.append(
            f"block_dims={block_dims}")
        if block_alphas is not None and block_alphas != -1 and block_alphas != "": args.network_args.append(
            f"block_alphas={block_alphas}")
        if conv_block_dims is not None and conv_block_dims != -1 and conv_block_dims != "": args.network_args.append(
            f"conv_block_dims={conv_block_dims}")
        if conv_block_alphas is not None and conv_block_alphas != -1 and conv_block_alphas != "": args.network_args.append(
            f"conv_block_alphas={conv_block_alphas}")
    if enable_block_weights:
        if down_lr_weight is not None and down_lr_weight != -1 and down_lr_weight != "": args.network_args.append(
            f"down_lr_weight={down_lr_weight}")
        if up_lr_weight is not None and up_lr_weight != -1 and up_lr_weight != "": args.network_args.append(
            f"up_lr_weight={up_lr_weight}")
        if mid_lr_weight is not None and mid_lr_weight != -1 and mid_lr_weight != "": args.network_args.append(
            f"mid_lr_weight={mid_lr_weight}")
        if block_lr_zero_threshold is not None and block_lr_zero_threshold != -1 and block_lr_zero_threshold != "": args.network_args.append(
            f"block_lr_zero_threshold={block_lr_zero_threshold}")

    if enable_preview:
        args.sample_prompts = sample_prompts if sample_prompts != "" and sample_prompts != -1 else None
        args.sample_sampler = sample_sampler if sample_sampler != "" and sample_sampler != -1 else None
        args.sample_every_n_epochs = sample_every_n_epochs if sample_every_n_epochs != "" and sample_every_n_epochs != -1 else None

    args.caption_extension = caption_extension if caption_extension != "" and caption_extension != -1 else None
    args.shuffle_caption = shuffle_caption if shuffle_caption != "" and shuffle_caption != -1 else None
    args.token_warmup_min = token_warmup_min if token_warmup_min != "" and token_warmup_min != -1 else None
    args.token_warmup_step = token_warmup_step if token_warmup_step != "" and token_warmup_step != -1 else None
    args.keep_tokens = keep_tokens if keep_tokens != "" and keep_tokens != -1 else None
    args.weighted_captions = weighted_captions if weighted_captions != "" and weighted_captions != -1 else None

    args.max_token_length = max_token_length if max_token_length != "" and max_token_length != -1 else None
    args.caption_dropout_rate = caption_dropout_rate if caption_dropout_rate != "" and caption_dropout_rate != -1 else None
    args.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs if caption_dropout_every_n_epochs != "" and caption_dropout_every_n_epochs != -1 else None
    args.caption_tag_dropout_rate = caption_tag_dropout_rate if caption_tag_dropout_rate != "" and caption_tag_dropout_rate != -1 else None

    args.prior_loss_weight = prior_loss_weight if prior_loss_weight != "" and prior_loss_weight != -1 else None
    args.min_snr_gamma = min_snr_gamma if min_snr_gamma != "" and min_snr_gamma != -1 else None
    args.noise_offset = noise_offset if noise_offset != "" and noise_offset != -1 else None
    args.multires_noise_iterations = multires_noise_iterations if multires_noise_iterations != "" and multires_noise_iterations != -1 else None
    args.multires_noise_discount = multires_noise_discount if multires_noise_discount != "" and multires_noise_discount != -1 else None
    args.gradient_checkpointing = gradient_checkpointing if gradient_checkpointing != "" and gradient_checkpointing != -1 else None
    args.gradient_accumulation_steps = gradient_accumulation_steps if gradient_accumulation_steps != "" and gradient_accumulation_steps != -1 else None
    args.mixed_precision = mixed_precision if mixed_precision != "" and mixed_precision != -1 else None
    if noise_offset is not None:
        args.adaptive_noise_scale = adaptive_noise_scale if adaptive_noise_scale != "" and adaptive_noise_scale != -1 else None

    args.xformers = xformers
    args.lowram = lowram
    args.cache_latents = cache_latents
    args.cache_latents_to_disk = cache_latents_to_disk
    args.persistent_data_loader_workers = persistent_data_loader_workers
    args.save_precision = save_precision

    args.config_file = config_file if config_file != "" and config_file != -1 else None
    args.output_config = output_config if output_config != "" and output_config != -1 else None
    if output_config is not None and config_file is not None:
        if config_file.endswith(".json"):
            args = train_util.read_config_from_json(args, parser)
        elif config_file.endswith(".toml"):
            args = train_util.read_config_from_file(args, parser)
    # print("network_args:",args.network_args)
    return train(args, callback)


if __name__ == "__main__":
    # parser = setup_parser()
    #
    # args = parser.parse_args()
    # args = train_util.read_config_from_file(args, parser)
    test_with_file = False
    if test_with_file:
        config_file_path = r"E:\qll\sd-scripts\test_config.toml"
        train_with_file(config_file_path, train_callback)
    else:
        train_with_params(

            pretrained_model_name_or_path=r"E:\qll\models\chilloutmix_NiPrunedFp32Fix.safetensors",
            network_weights="",  # "output/y1s1_100v3.safetensors",
            output_name="xhx-lion-0.0002",
            save_model_as="safetensors",
            v2=False,
            v_parameterization=False,
            output_dir="./output",
            logging_dir="./logs",
            save_every_n_epochs=2,
            save_last_n_epochs=10,
            save_precision=None,
            trigger_words=["xhx"],
            reg_tokens=[""],
            list_train_data_dir=[r"E:\qll\pics\hanfu-512x768-tags"],
            list_reg_data_dir=[""],
            max_token_length=75,  # max token length of text encoder (default for 75, 150 or 225)
            num_repeats=["8"],
            list_reg_repeats=None,  # ["8"]
            batch_size=1,
            resolution="512,512",  # 64的倍数
            cache_latents=False,
            # cache latents to main memory to reduce VRAM usage (augmentations must be disabled)
            cache_latents_to_disk=False,
            # cache latents to disk to reduce VRAM usage (augmentations must be disabled)
            enable_bucket=False,  # enable buckets for multi aspect ratio training
            min_bucket_reso=256,  # 范围自己定，minimum resolution for buckets
            max_bucket_reso=1024,  # 范围自己定，maximum resolution for buckets
            bucket_reso_steps=64,  # 秋叶版没有这个,steps of resolution for buckets, divisible by 8 is recommended
            bucket_no_upscale=False,  # 秋叶版没有这个,make bucket for each image without upscaling
            token_warmup_min=1,  # 秋叶版没有这个,start learning at N tags (token means comma separated strinfloatgs)
            token_warmup_step=0,  # 秋叶版没有这个,tag length reaches maximum on N steps (or N*max_train_steps if N<1)

            caption_extension=".txt",
            caption_dropout_rate=0.0,  # Rate out dropout caption(0.0~1.0)
            caption_dropout_every_n_epochs=0,  # Dropout all captions every N epochs
            caption_tag_dropout_rate=0.0,  # Rate out dropout comma separated tokens(0.0~1.0)
            shuffle_caption=False,  # shuffle comma-separated caption
            weighted_captions=False,  # 使用带权重的 token，不推荐与 shuffle_caption 一同开启
            keep_tokens=0,
            # keep heading N tokens when shuffling caption tokens (token means comma separated strings)
            color_aug=False,  # 秋叶版没有这个,enable weak color augmentation
            flip_aug=False,  # 秋叶版没有这个,enable horizontal flip augmentation
            face_crop_aug_range=None,  # 1.0,2.0,4.0
            # 秋叶版没有这个,enable face-centered crop augmentation and its range (e.g. 2.0,4.0)
            random_crop=False,
            # 秋叶版没有这个,enable random crop (for style training in face-centered crop augmentation)

            lowram=True,
            # enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle)
            mem_eff_attn=False,  # use memory efficient attention for CrossAttention
            xformers=True,  # 如果mem_eff_attn为True则xformers设置无效
            vae=None,  # 比如：c:\vae.pt, 秋叶版没有这个,path to checkpoint of vae to replace
            max_data_loader_n_workers=8,
            # 秋叶版没有这个,max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading)
            persistent_data_loader_workers=True,
            # persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory)

            max_train_steps=1600,  # 秋叶版没有这个,
            epoch=10,  # 整数，随便填
            gradient_checkpointing=True,
            gradient_accumulation_steps=1,
            # 整数，随便填, Number of updates steps to accumulate before performing a backward/update pass
            mixed_precision="no",  # 是否使用混精度
            full_fp16=True,  # fp16 training including gradients

            enable_preview=False,  # 和下面这几个参数一起的
            sample_prompts=None,  # 文件路径，比如c:\promts.txt,file for prompts to generate sample images
            sample_sampler="ddim",
            # ["ddim","pndm","lms","euler","euler_a","heun","dpm_2","dpm_2_a","dpmsolver","dpmsolver++","dpmsingle","k_lms","k_euler","k_euler_a","k_dpm_2","k_dpm_2_a",]
            sample_every_n_epochs=None,  # 1,2,3,4,5.....

            network_module="networks.lora",
            network_train_unet_only=False,
            network_train_text_encoder_only=False,
            network_dim=32,  # 4的倍数，<=256
            network_alpha=16,  # 小于等于network_dim,可以不是4的倍数
            clip_skip=1,  # 0-12

            # network额外参数
            conv_dim=None,  # 默认为None，可以填4的倍数，类似于network_dim,
            # lycoris才有，# 4的倍数, 适用于lora,dylora。如果是dylora,则"conv_dim must be same as network_dim",
            conv_alpha=None,  # 默认为None，可以填比con_dim小的整数，类似于network_alpha； lycoris才有，<=conv_dim, 适用于lora,dylora
            unit=8,  # 秋叶版没有
            dropout=0,  # dropout 概率, 0 为不使用 dropout, 越大则 dropout 越多，推荐 0~0.5， LoHa/LoKr/(IA)^3暂时不支持
            algo='lora',  # 可选['lora','loha','lokr','ia3']

            enable_block_weights=False,  # 让下面几个参数有效
            block_dims=None,  # lora,  类似于network_dim,
            block_alphas=None,  # lora,默认为None，可以填比con_dim小的整数，类似于network_alpha
            conv_block_dims=None,  # lora,  类似于network_dim,
            conv_block_alphas=None,  # lora,默认为None，可以填比con_dim小的整数，类似于network_alpha
            down_lr_weight=None,  # lora, 12位的float List，例如[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            mid_lr_weight=None,  # lora, 1位float,例如 1.0；
            up_lr_weight=None,  # lora, 12位的float List，例如[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            block_lr_zero_threshold=0.0,  # float型，分层学习率置 0 阈值

            optimizer_type="AdamW8bit",
            # AdamW (default), AdamW8bit, Lion8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation(DAdaptAdam), DAdaptAdaGrad, DAdaptAdan, DAdaptSGD, AdaFactor
            weight_decay=None,  # optimizer_args,优化器内部的参数，权重衰减系数，不建议随便改
            betas=None,  # optimizer_args,优化器内部的参数，不建议随便改

            max_grad_norm=1.0,  # Max gradient norm, 0 for no clipping

            learning_rate=0.0001,
            unet_lr=0.0001,
            text_encoder_lr=0.00001,
            lr_scheduler="cosine_with_restarts",
            # linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor
            lr_scheduler_num_cycles=1,  # Number of restarts for cosine scheduler with restarts
            lr_warmup_steps=0,  # Number of steps for the warmup in the lr scheduler
            lr_scheduler_power=1,  # Polynomial power for polynomial scheduler

            seed=1,
            prior_loss_weight=1.0,  # loss weight for regularization images
            min_snr_gamma=None,  # float型，比如5.0，最小信噪比伽马值，如果启用推荐为 5
            noise_offset=None,
            # float型，0.1左右,enable noise offset with this value (if enabled, around 0.1 is recommended)
            adaptive_noise_scale=None,  # float型， 1.0
            # 与noise_offset配套使用；add `latent mean absolute value * this value` to noise_offset (disabled if None, default)
            multires_noise_iterations=None,  # 整数，多分辨率（金字塔）噪声迭代次数 推荐 6-10。无法与 noise_offset 一同启用。
            multires_noise_discount=0.3,  # 多分辨率（金字塔）噪声迭代次数 推荐 6-10。无法与 noise_offset 一同启用。

            config_file="test_config.toml",  # using .toml instead of args to pass hyperparameter
            output_config=False,  # output command line args to given .toml file
        )
    # train(args)
# python train_network_qll.py --pretrained_model_name_or_path "/data/qll/qianzai_ai_draw/v1-5-pruned-emaonly.ckpt" --train_data_dir "/data/qll/lora_pictures/train_ironman" --output_name im2 --resolution 512 --network_module "networks.lora" --network_dim 32 --xformers  --caption_extension ".txt" --prior_loss_weight 1 --output_dir "./output" --logging_dir "./logs" --repeats_times "20" --class_tokens iiiiimqll --output_name iiiiimqll --list_train_data_dir "/data/qll/lora_pictures/train_ironman/10_immmmmman"
