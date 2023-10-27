# cond_imageをU-Netのforwardで渡すバージョンのControlNet-LLLite検証用学習コード
# training code for ControlNet-LLLite with passing cond_image to U-Net's forward

import argparse
import gc
import json
import math
import os
import random
import time
from multiprocessing import Value
from types import SimpleNamespace
import toml

from tqdm import tqdm
import torch
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from library.ipex import ipex_init
        ipex_init()
except Exception:
    pass
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate.utils import set_seed
import accelerate
from diffusers import DDPMScheduler, ControlNetModel
from safetensors.torch import load_file
from library import sai_model_spec, sdxl_model_util, sdxl_original_unet, sdxl_train_util

import library.model_util as model_util
import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    add_v_prediction_like_loss,
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    pyramid_noise_like,
    apply_noise_offset,
    scale_v_prediction_loss_like_noise_prediction,
)
import networks.control_net_lllite_for_train as control_net_lllite_for_train


# TODO 他のスクリプトと共通化する
def generate_step_logs(args: argparse.Namespace, current_loss, avr_loss, lr_scheduler):
    logs = {
        "loss/current": current_loss,
        "loss/average": avr_loss,
        "lr": lr_scheduler.get_last_lr()[0],
    }

    if args.optimizer_type.lower().startswith("DAdapt".lower()):
        logs["lr/d*lr"] = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]

    return logs


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)

    cache_latents = args.cache_latents
    use_user_config = args.dataset_config is not None

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    # データセットを準備する
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(False, False, True, True))
    if use_user_config:
        print(f"Load dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        ignored = ["train_data_dir", "conditioning_data_dir"]
        if any(getattr(args, attr) is not None for attr in ignored):
            print(
                "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                    ", ".join(ignored)
                )
            )
    else:
        user_config = {
            "datasets": [
                {
                    "subsets": config_util.generate_controlnet_subsets_config_by_subdirs(
                        args.train_data_dir,
                        args.conditioning_data_dir,
                        args.caption_extension,
                    )
                }
            ]
        }

    blueprint = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer1, tokenizer2])
    train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

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
    else:
        print("WARNING: random_crop is not supported yet for ControlNet training / ControlNetの学習ではrandom_cropはまだサポートされていません")

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    print("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(
                vae,
                args.vae_batch_size,
                args.cache_latents_to_disk,
                accelerator.is_main_process,
            )
        vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        accelerator.wait_for_everyone()

    # TextEncoderの出力をキャッシュする
    if args.cache_text_encoder_outputs:
        # Text Encodes are eval and no grad
        with torch.no_grad():
            train_dataset_group.cache_text_encoder_outputs(
                (tokenizer1, tokenizer2),
                (text_encoder1, text_encoder2),
                accelerator.device,
                None,
                args.cache_text_encoder_outputs_to_disk,
                accelerator.is_main_process,
            )
        accelerator.wait_for_everyone()

    # prepare ControlNet-LLLite
    control_net_lllite_for_train.replace_unet_linear_and_conv2d()

    if args.network_weights is not None:
        accelerator.print(f"initialize U-Net with ControlNet-LLLite")
        with accelerate.init_empty_weights():
            unet_lllite = control_net_lllite_for_train.SdxlUNet2DConditionModelControlNetLLLite()
        unet_lllite.to(accelerator.device, dtype=weight_dtype)

        unet_sd = unet.state_dict()
        info = unet_lllite.load_lllite_weights(args.network_weights, unet_sd)
        accelerator.print(f"load ControlNet-LLLite weights from {args.network_weights}: {info}")
    else:
        # cosumes large memory, so send to GPU before creating the LLLite model
        accelerator.print("sending U-Net to GPU")
        unet.to(accelerator.device, dtype=weight_dtype)
        unet_sd = unet.state_dict()

        # init LLLite weights
        accelerator.print(f"initialize U-Net with ControlNet-LLLite")

        if args.lowram:
            with accelerate.init_on_device(accelerator.device):
                unet_lllite = control_net_lllite_for_train.SdxlUNet2DConditionModelControlNetLLLite()
        else:
            unet_lllite = control_net_lllite_for_train.SdxlUNet2DConditionModelControlNetLLLite()
        unet_lllite.to(weight_dtype)

        info = unet_lllite.load_lllite_weights(None, unet_sd)
        accelerator.print(f"init U-Net with ControlNet-LLLite weights: {info}")
    del unet_sd, unet

    unet: control_net_lllite_for_train.SdxlUNet2DConditionModelControlNetLLLite = unet_lllite
    del unet_lllite

    unet.apply_lllite(args.cond_emb_dim, args.network_dim, args.network_dropout)

    # モデルに xformers とか memory efficient attention を組み込む
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    trainable_params = list(unet.prepare_params())
    print(f"trainable params count: {len(trainable_params)}")
    print(f"number of trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad)}")

    _, _, optimizer = train_util.get_optimizer(args, trainable_params)

    # dataloaderを準備する
    # DataLoaderのプロセス数：0はメインプロセスになる
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    # if args.full_fp16:
    #     assert (
    #         args.mixed_precision == "fp16"
    #     ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
    #     accelerator.print("enable full fp16 training.")
    #     unet.to(weight_dtype)
    # elif args.full_bf16:
    #     assert (
    #         args.mixed_precision == "bf16"
    #     ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
    #     accelerator.print("enable full bf16 training.")
    #     unet.to(weight_dtype)

    unet.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    # transform DDP after prepare (train_network here only)
    unet = train_util.transform_models_if_DDP([unet])[0]

    if args.gradient_checkpointing:
        unet.train()  # according to TI example in Diffusers, train is required -> これオリジナルのU-Netしたので本当は外せる
    else:
        unet.eval()

    # TextEncoderの出力をキャッシュするときにはCPUへ移動する
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # make sure Text Encoders are on GPU
        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

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
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
    # print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "lllite_control_net_train" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs
        )

    loss_list = []
    loss_total = 0.0
    del train_dataset_group

    # function for saving/removing
    def save_model(
        ckpt_name,
        unwrapped_nw: control_net_lllite_for_train.SdxlUNet2DConditionModelControlNetLLLite,
        steps,
        epoch_no,
        force_sync_upload=False,
    ):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        sai_metadata = train_util.get_sai_model_spec(None, args, True, True, False)
        sai_metadata["modelspec.architecture"] = sai_model_spec.ARCH_SD_XL_V1_BASE + "/control-net-lllite"

        unwrapped_nw.save_lllite_weights(ckpt_file, save_dtype, sai_metadata)
        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def remove_model(old_ckpt_name):
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

    # training loop
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device)
                    else:
                        # latentに変換
                        latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample()

                        # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                    latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
                    input_ids1 = batch["input_ids"]
                    input_ids2 = batch["input_ids2"]
                    with torch.no_grad():
                        # Get the text embedding for conditioning
                        input_ids1 = input_ids1.to(accelerator.device)
                        input_ids2 = input_ids2.to(accelerator.device)
                        encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                            args.max_token_length,
                            input_ids1,
                            input_ids2,
                            tokenizer1,
                            tokenizer2,
                            text_encoder1,
                            text_encoder2,
                            None if not args.full_fp16 else weight_dtype,
                        )
                else:
                    encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
                    encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
                    pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

                # get size embeddings
                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                # concat embeddings
                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                controlnet_image = batch["conditioning_images"].to(dtype=weight_dtype)

                with accelerator.autocast():
                    # conditioning imageをControlNetに渡す / pass conditioning image to ControlNet
                    # 内部でcond_embに変換される / it will be converted to cond_emb inside

                    # それらの値を使いつつ、U-Netでノイズを予測する / predict noise with U-Net using those values
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding, controlnet_image)

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
                if args.scale_v_pred_loss_like_noise_pred:
                    loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                if args.v_pred_like_loss:
                    loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)

                loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = unet.get_trainable_params()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # sdxl_train_util.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                        save_model(ckpt_name, accelerator.unwrap_model(unet), global_step, epoch)

                        if args.save_state:
                            train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                        remove_step_no = train_util.get_remove_step_no(args, global_step)
                        if remove_step_no is not None:
                            remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
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
                save_model(ckpt_name, accelerator.unwrap_model(unet), global_step, epoch + 1)

                remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                if remove_epoch_no is not None:
                    remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                    remove_model(remove_ckpt_name)

                if args.save_state:
                    train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

        # self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

        # end of epoch

    if is_main_process:
        unet = accelerator.unwrap_model(unet)

    accelerator.end_training()

    if is_main_process and args.save_state:
        train_util.save_state_on_train_end(args, accelerator)

    if is_main_process:
        ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
        save_model(ckpt_name, unet, global_step, num_train_epochs, force_sync_upload=True)

        print("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, False, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )
    parser.add_argument("--cond_emb_dim", type=int, default=None, help="conditioning embedding dimension / 条件付け埋め込みの次元数")
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_dim", type=int, default=None, help="network dimensions (rank) / モジュールの次元数")
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--conditioning_data_dir",
        type=str,
        default=None,
        help="conditioning data directory / 条件付けデータのディレクトリ",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    return parser


if __name__ == "__main__":
    # sdxl_original_unet.USE_REENTRANT = False

    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(args)
