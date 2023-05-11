# DreamBooth training
# XXX dropped option: fine_tune

import gc
import time
import argparse
import itertools
import math
import os
import toml
from multiprocessing import Value

from tqdm import tqdm
import torch
from accelerate.utils import set_seed
import diffusers
from diffusers import DDPMScheduler

import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import apply_snr_weight, get_weighted_text_embeddings, pyramid_noise_like, apply_noise_offset


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, False)

    cache_latents = args.cache_latents

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenizer = train_util.load_tokenizer(args)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, False, True))
    if args.dataset_config is not None:
        print(f"Load dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        ignored = ["train_data_dir", "reg_data_dir"]
        if any(getattr(args, attr) is not None for attr in ignored):
            print(
                "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                    ", ".join(ignored)
                )
            )
    else:
        user_config = {
            "datasets": [
                {"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)}
            ]
        }

    blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
    train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)

    if args.no_token_padding:
        train_dataset_group.disable_token_padding()

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group)
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    # acceleratorを準備する
    print("prepare accelerator")

    if args.gradient_accumulation_steps > 1:
        print(
            f"gradient_accumulation_steps is {args.gradient_accumulation_steps}. accelerate does not support gradient_accumulation_steps when training multiple models (U-Net and Text Encoder), so something might be wrong"
        )
        print(
            f"gradient_accumulation_stepsが{args.gradient_accumulation_steps}に設定されています。accelerateは複数モデル（U-NetおよびText Encoder）の学習時にgradient_accumulation_stepsをサポートしていないため結果は未知数です"
        )

    accelerator, unwrap_model = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # モデルを読み込む
    text_encoder, vae, unet, load_stable_diffusion_format = train_util.load_target_model(args, weight_dtype, accelerator)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())

    # モデルに xformers とか memory efficient attention を組み込む
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
        vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        accelerator.wait_for_everyone()

    # 学習を準備する：モデルを適切な状態にする
    train_text_encoder = args.stop_text_encoder_training is None or args.stop_text_encoder_training >= 0
    unet.requires_grad_(True)  # 念のため追加
    text_encoder.requires_grad_(train_text_encoder)
    if not train_text_encoder:
        print("Text Encoder is not trained.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    # 学習に必要なクラスを準備する
    print("prepare optimizer, data loader etc.")
    if train_text_encoder:
        trainable_params = itertools.chain(unet.parameters(), text_encoder.parameters())
    else:
        trainable_params = unet.parameters()

    _, _, optimizer = train_util.get_optimizer(args, trainable_params)

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
        print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    if args.stop_text_encoder_training is None:
        args.stop_text_encoder_training = args.max_train_steps + 1  # do not stop until end

    # lr schedulerを用意する TODO gradient_accumulation_stepsの扱いが何かおかしいかもしれない。後で確認する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    # transform DDP after prepare
    text_encoder, unet = train_util.transform_if_model_is_DDP(text_encoder, unet)

    if not train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)  # to avoid 'cpu' vs 'cuda' error

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
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    print("running training / 学習開始")
    print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
    print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
    print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    print(f"  num epochs / epoch数: {num_train_epochs}")
    print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
    print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
    print(f"  gradient ccumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth" if args.log_tracker_name is None else args.log_tracker_name)

    loss_list = []
    loss_total = 0.0
    for epoch in range(num_train_epochs):
        print(f"epoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        # 指定したステップ数までText Encoderを学習する：epoch最初の状態
        unet.train()
        # train==True is required to enable gradient_checkpointing
        if args.gradient_checkpointing or global_step < args.stop_text_encoder_training:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            # 指定したステップ数でText Encoderの学習を止める
            if global_step == args.stop_text_encoder_training:
                print(f"stop text encoder training at step {global_step}")
                if not args.gradient_checkpointing:
                    text_encoder.train(False)
                text_encoder.requires_grad_(False)

            with accelerator.accumulate(unet):
                with torch.no_grad():
                    # latentに変換
                    if cache_latents:
                        latents = batch["latents"].to(accelerator.device)
                    else:
                        latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215
                b_size = latents.shape[0]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents, device=latents.device)
                if args.noise_offset:
                    noise = apply_noise_offset(latents, noise, args.noise_offset, args.adaptive_noise_scale)
                elif args.multires_noise_iterations:
                    noise = pyramid_noise_like(noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount)

                # Get the text embedding for conditioning
                with torch.set_grad_enabled(global_step < args.stop_text_encoder_training):
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
                        encoder_hidden_states = train_util.get_hidden_states(
                            args, input_ids, tokenizer, text_encoder, None if not args.full_fp16 else weight_dtype
                        )

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
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
                    if train_text_encoder:
                        params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters())
                    else:
                        params_to_clip = unet.parameters()
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
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            unwrap_model(text_encoder),
                            unwrap_model(unet),
                            vae,
                        )

            current_loss = loss.detach().item()
            if args.logging_dir is not None:
                logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
                if args.optimizer_type.lower().startswith("DAdapt".lower()):  # tracking d*lr value
                    logs["lr/d*lr"] = (
                        lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"]
                    )
                accelerator.log(logs, step=global_step)

            if epoch == 0:
                loss_list.append(current_loss)
            else:
                loss_total -= loss_list[step]
                loss_list[step] = current_loss
            loss_total += current_loss
            avr_loss = loss_total / len(loss_list)
            logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_total / len(loss_list)}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                # checking for saving is in util
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    unwrap_model(text_encoder),
                    unwrap_model(unet),
                    vae,
                )

        train_util.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

    is_main_process = accelerator.is_main_process
    if is_main_process:
        unet = unwrap_model(unet)
        text_encoder = unwrap_model(text_encoder)

    accelerator.end_training()

    if args.save_state and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        train_util.save_sd_model_on_train_end(
            args, src_path, save_stable_diffusion_format, use_safetensors, save_dtype, epoch, global_step, text_encoder, unet, vae
        )
        print("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, False, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--no_token_padding",
        action="store_true",
        help="disable token padding (same as Diffuser's DreamBooth) / トークンのpaddingを無効にする（Diffusers版DreamBoothと同じ動作）",
    )
    parser.add_argument(
        "--stop_text_encoder_training",
        type=int,
        default=None,
        help="steps to stop text encoder training, -1 for no training / Text Encoderの学習を止めるステップ数、-1で最初から学習しない",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(args)
