import datetime
import json
import os

saved_params_shared = {"model_name", "model_hash", "initial_step", "num_of_dataset_images", "learn_rate", "batch_size", "clip_grad_mode", "clip_grad_value", "gradient_step", "data_root", "log_directory", "training_width", "training_height", "steps", "create_image_every", "template_file", "gradient_step", "latent_sampling_method"}
saved_params_ti = {"embedding_name", "num_vectors_per_token", "save_embedding_every", "save_image_with_stored_embedding"}
saved_params_hypernet = {"hypernetwork_name", "layer_structure", "activation_func", "weight_init", "add_layer_norm", "use_dropout", "save_hypernetwork_every"}
saved_params_all = saved_params_shared | saved_params_ti | saved_params_hypernet
saved_params_previews = {"preview_prompt", "preview_negative_prompt", "preview_steps", "preview_sampler_index", "preview_cfg_scale", "preview_seed", "preview_width", "preview_height"}


def save_settings_to_file(log_directory, all_params):
    now = datetime.datetime.now()
    params = {"datetime": now.strftime("%Y-%m-%d %H:%M:%S")}

    keys = saved_params_all
    if all_params.get('preview_from_txt2img'):
        keys = keys | saved_params_previews

    params.update({k: v for k, v in all_params.items() if k in keys})

    filename = f'settings-{now.strftime("%Y-%m-%d-%H-%M-%S")}.json'
    with open(os.path.join(log_directory, filename), "w") as file:
        json.dump(params, file, indent=4)
