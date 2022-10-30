import json
import os

from modules import paths, images


class TrainConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = None
        self.model_name = None
        self.scheduler = None
        self.src = None
        self.total_steps = None
        self.__dict__ = self

    def create_new(self, name, scheduler, src, total_steps):
        name = images.sanitize_filename_part(name, True)
        self.model_name = name
        self.scheduler = scheduler
        self.src = src
        self.total_steps = total_steps
        return self

    def from_ui(self,
                model_name,
                initialization_text,
                classification_text,
                learn_rate,
                dataset_directory,
                classification_directory,
                steps,
                save_preview_every,
                save_embedding_every,
                num_class_images,
                use_cpu,
                train_text_encoder,
                use_adam,
                center_crop,
                grad_check,
                scale_lr,
                mixed_precision,
                scheduler,
                resolution,
                prior_loss_weight,
                num_train_epochs,
                adam_beta1,
                adam_beta2,
                adam_weight_decay,
                adam_epsilon,
                max_grad_norm,
                batch_size,
                class_batch_size,
                seed,
                grad_acc_steps,
                warmup_steps):
        """
        Update config from UI
        Args:
            model_name:
            initialization_text:
            classification_text:
            learn_rate:
            dataset_directory:
            classification_directory:
            steps:
            save_preview_every:
            save_embedding_every:
            num_class_images:
            use_cpu:
            train_text_encoder:
            use_adam:
            center_crop:
            grad_check:
            scale_lr:
            mixed_precision:
            scheduler:
            resolution:
            prior_loss_weight:
            num_train_epochs:
            adam_beta1:
            adam_beta2:
            adam_weight_decay:
            adam_epsilon:
            max_grad_norm:
            batch_size:
            class_batch_size:
            seed:
            grad_acc_steps:
            warmup_steps:

        Returns:

        """
        model_name = images.sanitize_filename_part(model_name, True)
        data = {"model_name": model_name, "instance_prompt": initialization_text,
                "class_prompt": classification_text, "learn_rate": learn_rate,
                "instance_data_dir": dataset_directory, "class_data_dir": classification_directory,
                "steps": steps, "save_preview_every": save_preview_every, "save_embedding_every": save_embedding_every,
                "num_class_images": num_class_images, "use_cpu": use_cpu, "train_text_encoder": train_text_encoder,
                "use_adam": use_adam, "center_crop": center_crop, "gradient_checkpointing": grad_check, "scale_lr": scale_lr,
                "mixed_precision": mixed_precision, "scheduler": scheduler, "resolution": resolution,
                "prior_loss_weight": prior_loss_weight, "num_train_epochs": num_train_epochs, "adam_beta1": adam_beta1,
                "adam_beta2": adam_beta2, "adam_weight_decay": adam_weight_decay, "adam_epsilon": adam_epsilon,
                "max_grad_norm": max_grad_norm, "batch_size": batch_size, "class_batch_size": class_batch_size,
                "seed": seed, "grad_acc_steps": grad_acc_steps, "warmup_steps": warmup_steps}
        for key in data:
            self.__dict__[key] = data[key]
        return self.__dict__

    def from_file(self, model_name):
        """
        Load config data from UI
        Args:
            model_name: The config to load

        Returns: Dict

        """
        model_name = images.sanitize_filename_part(model_name, True)
        model_path = paths.models_path
        config_file = os.path.join(model_path, "dreambooth", model_name, "db_config.json")
        try:
            with open(config_file, 'r') as openfile:
                config = json.load(openfile)
                for key in config:
                    self.__dict__[key] = config[key]
        except Exception as e:
            print(f"Exception loading config: {e}")
            return None
            pass
        return self.__dict__

    def save(self):
        """
        Save the config file3
        """
        model_path = paths.models_path
        config_file = os.path.join(model_path, "dreambooth", self.__dict__["model_name"], "db_config.json")
        config = json.dumps(self.__dict__)
        with open(config_file, "w") as outfile:
            outfile.write(config)
