import os
import huggingface_hub as hf
from modules import shared, processing, sd_models, devices


def photo_maker(p: processing.StableDiffusionProcessing, input_images, trigger, strength, start): # pylint: disable=arguments-differ
    from modules.face.photomaker_model import PhotoMakerStableDiffusionXLPipeline

    # prepare pipeline
    if len(input_images) == 0:
        shared.log.warning('PhotoMaker: no input images')
        return None

    c = shared.sd_model.__class__.__name__ if shared.sd_model is not None else ''
    if c != 'StableDiffusionXLPipeline':
        shared.log.warning(f'PhotoMaker invalid base model: current={c} required=StableDiffusionXLPipeline')
        return None

    # validate prompt
    trigger_ids = shared.sd_model.tokenizer.encode(trigger) + shared.sd_model.tokenizer_2.encode(trigger)
    prompt_ids1 = shared.sd_model.tokenizer.encode(p.all_prompts[0])
    prompt_ids2 = shared.sd_model.tokenizer_2.encode(p.all_prompts[0])
    for t in trigger_ids:
        if prompt_ids1.count(t) != 1:
            shared.log.error(f'PhotoMaker: trigger word not matched in prompt: {trigger} ids={trigger_ids} prompt={p.all_prompts[0]} ids={prompt_ids1}')
            return None
        if prompt_ids2.count(t) != 1:
            shared.log.error(f'PhotoMaker: trigger word not matched in prompt: {trigger} ids={trigger_ids} prompt={p.all_prompts[0]} ids={prompt_ids1}')
            return None

    # create new pipeline
    orig_pipeline = shared.sd_model # backup current pipeline definition
    shared.sd_model = PhotoMakerStableDiffusionXLPipeline(
        vae = shared.sd_model.vae,
        text_encoder=shared.sd_model.text_encoder,
        text_encoder_2=shared.sd_model.text_encoder_2,
        tokenizer=shared.sd_model.tokenizer,
        tokenizer_2=shared.sd_model.tokenizer_2,
        unet=shared.sd_model.unet,
        scheduler=shared.sd_model.scheduler,
        force_zeros_for_empty_prompt=shared.opts.diffusers_force_zeros,
    )
    sd_models.copy_diffuser_options(shared.sd_model, orig_pipeline) # copy options from original pipeline
    sd_models.set_diffuser_options(shared.sd_model) # set all model options such as fp16, offload, etc.
    sd_models.move_model(shared.sd_model, devices.device) # move pipeline to device
    shared.sd_model.to(dtype=devices.dtype)

    orig_prompt_attention = shared.opts.prompt_attention
    shared.opts.data['prompt_attention'] = 'Fixed attention' # otherwise need to deal with class_tokens_mask
    p.task_args['input_id_images'] = input_images
    p.task_args['start_merge_step'] = int(start * p.steps)
    p.task_args['prompt'] = p.all_prompts[0] # override all logic

    photomaker_path = hf.hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model", cache_dir=shared.opts.diffusers_dir)
    shared.log.debug(f'PhotoMaker: model={photomaker_path} images={len(input_images)} trigger={trigger} args={p.task_args}')

    # load photomaker adapter
    shared.sd_model.load_photomaker_adapter(
        os.path.dirname(photomaker_path),
        subfolder="",
        weight_name=os.path.basename(photomaker_path),
        trigger_word=trigger
    )
    shared.sd_model.set_adapters(["photomaker"], adapter_weights=[strength])

    # run processing
    processed: processing.Processed = processing.process_images(p)
    p.extra_generation_params['PhotoMaker'] = f'{strength}'

    # restore original pipeline
    shared.opts.data['prompt_attention'] = orig_prompt_attention
    shared.sd_model = orig_pipeline
    return processed
