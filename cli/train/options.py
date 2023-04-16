from util import Map

embedding = Map({
    "id_task": 0,
    "embedding_name": "",
    "learn_rate": -1,
    "batch_size": 1,
    "steps": 500,
    "data_root": "",
    "log_directory": "train/log",
    "template_filename": "subject_filewords.txt",
    "gradient_step": 20,
    "training_width": 512,
    "training_height": 512,
    "shuffle_tags": False,
    "tag_drop_out": 0,
    "clip_grad_mode": "disabled",
    "clip_grad_value": "0.1",
    "latent_sampling_method": "deterministic",
    "create_image_every": 0,
    "save_embedding_every": 0,
    "save_image_with_stored_embedding": False,
    "preview_from_txt2img": False,
    "preview_prompt": "",
    "preview_negative_prompt": "blurry, duplicate, ugly, deformed, low res, watermark, text",
    "preview_steps": 20,
    "preview_sampler_index": 0,
    "preview_cfg_scale": 6,
    "preview_seed": -1,
    "preview_width": 512,
    "preview_height": 512,
    "varsize": False,
    "use_weight": False,
})

lora = Map({
    "bucket_no_upscale": False,
    "bucket_reso_steps": 64,
    "cache_latents": True,
    "caption_dropout_every_n_epochs": None,
    "caption_dropout_rate": 0.0,
    "caption_extension": ".txt",
    "caption_extention": ".txt",
    "caption_tag_dropout_rate": 0.0,
    "clip_skip": None,
    "color_aug": False,
    "dataset_repeats": 1,
    "debug_dataset": False,
    "enable_bucket": False,
    "face_crop_aug_range": None,
    "flip_aug": False,
    "full_fp16": False,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": False,
    "in_json": "",
    "keep_tokens": None,
    "learning_rate": 5e-05,
    "log_prefix": None,
    "logging_dir": None,
    "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": 1,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 0,
    "max_bucket_reso": 1024,
    "max_data_loader_n_workers": 8,
    "max_grad_norm": 0.0,
    "max_token_length": None,
    "max_train_epochs": None,
    "max_train_steps": 2500,
    "mem_eff_attn": False,
    "min_bucket_reso": 256,
    "mixed_precision": "fp16",
    "network_alpha": 1.0,
    "network_args": None,
    "network_dim": 16,
    "network_module": "networks.lora",
    "network_train_text_encoder_only": False,
    "network_train_unet_only": False,
    "network_weights": None,
    "no_metadata": False,
    "output_dir": "",
    "output_name": "",
    "persistent_data_loader_workers": False,
    "pretrained_model_name_or_path": "",
    "prior_loss_weight": 1.0,
    "random_crop": False,
    "reg_data_dir": None,
    "resolution": "512,512",
    "resume": None,
    "save_every_n_epochs": None,
    "save_last_n_epochs_state": None,
    "save_last_n_epochs": None,
    "save_model_as": "ckpt",
    "save_n_epoch_ratio": None,
    "save_precision": "fp16",
    "save_state": False,
    "seed": 42,
    "shuffle_caption": False,
    "text_encoder_lr": 5e-05,
    "train_batch_size": 1,
    "train_data_dir": "",
    "training_comment": "mood-magic",
    "unet_lr": 1e-04,
    "use_8bit_adam": False,
    "v_parameterization": False,
    "v2": False,
    "vae": None,
    "xformers": False,
})

process = Map({
    # general settings, do not modify
    'format': '.jpg', # image format
    'target_size': 512, # target resolution
    'segmentation_model': 0, # segmentation model 0/general 1/landscape
    'segmentation_background': (192, 192, 192), # segmentation background color
    'blur_score': 1.8, # max score for face blur detection
    'blur_samplesize': 60, # sample size to use for blur detection
    'similarity_score': 0.8, # maximum similarity score before image is discarded
    'similarity_size': 64, # base similarity detection on reduced images
    'range_score': 0.15, # min score for face color dynamicrange detection
    # face processing settings
    'face_score': 0.7, # min face detection score
    'face_pad': 0.1, # pad face image percentage
    'face_model': 1, # which face model to use 0/close-up 1/standard
    # body processing settings
    'body_score': 0.9, # min body detection score
    'body_visibility': 0.5, # min visibility score for each detected body part
    'body_parts': 15, # min number of detected body parts with sufficient visibility
    'body_pad': 0.2,  # pad body image percentage
    'body_model': 2, # body model to use 0/low 1/medium 2/high
    # similarity detection settings
    # interrogate settings
    'interrogate': False, # interrogate images
    'interrogate_model': ['clip', 'deepdanbooru'], # interrogate models
    'tag_limit': 5, # number of tags to extract
    # validations
    # tbd
    'face_segmentation': False, # segmentation enabled
    'body_segmentation': False, # segmentation enabled
})
