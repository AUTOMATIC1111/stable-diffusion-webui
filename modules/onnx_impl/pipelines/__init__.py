import os
import json
import shutil
import tempfile
from abc import ABCMeta
from typing import Type, Tuple, List, Any, Dict
from packaging import version
import onnx
import torch
import diffusers
import onnxruntime as ort
import optimum.onnxruntime
from installer import log
from modules import shared
from modules.paths import sd_configs_path, models_path
from modules.sd_models import CheckpointInfo
from modules.processing import StableDiffusionProcessing
from modules.olive_script import config
from modules.onnx_impl import DynamicSessionOptions, TorchCompatibleModule, VAE, run_olive_workflow
from modules.onnx_impl.utils import extract_device, move_inference_session, check_diffusers_cache, check_pipeline_sdxl, check_cache_onnx, load_init_dict, load_submodel, load_submodels, patch_kwargs, load_pipeline, get_base_constructor, get_io_config
from modules.onnx_impl.execution_providers import ExecutionProvider, EP_TO_NAME, get_provider


SUBMODELS_SD = ("text_encoder", "unet", "vae_encoder", "vae_decoder",)
SUBMODELS_SDXL = ("text_encoder", "text_encoder_2", "unet", "vae_encoder", "vae_decoder",)
SUBMODELS_SDXL_REFINER = ("text_encoder_2", "unet", "vae_encoder", "vae_decoder",)


class PipelineBase(TorchCompatibleModule, diffusers.DiffusionPipeline, metaclass=ABCMeta):
    model_type: str
    sd_model_hash: str
    sd_checkpoint_info: CheckpointInfo
    sd_model_checkpoint: str

    def __init__(self): # pylint: disable=super-init-not-called
        self.model_type = self.__class__.__name__

    def to(self, *args, **kwargs):
        if self.__class__ == OnnxRawPipeline: # cannot move pipeline which is not preprocessed.
            return self

        expected_modules, _ = self._get_signature_keys(self)
        for name in expected_modules:
            if not hasattr(self, name):
                log.warning(f"Pipeline does not have module '{name}'.")
                continue

            module = getattr(self, name)

            if isinstance(module, optimum.onnxruntime.modeling_diffusion._ORTDiffusionModelPart): # pylint: disable=protected-access
                device = extract_device(args, kwargs)
                if device is None:
                    return self
                module.session = move_inference_session(module.session, device)

            if not isinstance(module, diffusers.OnnxRuntimeModel):
                continue

            try:
                setattr(self, name, module.to(*args, **kwargs))
                del module
            except Exception:
                log.debug(f"Component device/dtype conversion failed: module={name} args={args}, kwargs={kwargs}")
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **_): # pylint: disable=arguments-differ
        return OnnxRawPipeline(
            cls,
            pretrained_model_name_or_path,
        )

    @classmethod
    def from_single_file(cls, pretrained_model_name_or_path, **_):
        return OnnxRawPipeline(
            cls,
            pretrained_model_name_or_path,
        )

    @classmethod
    def from_ckpt(cls, pretrained_model_name_or_path, **_):
        return cls.from_single_file(pretrained_model_name_or_path)


class CallablePipelineBase(PipelineBase):
    vae: VAE

    def __init__(self):
        super().__init__()
        self.vae = VAE(self)


class OnnxRawPipeline(PipelineBase):
    config = {}
    _is_sdxl: bool
    is_refiner: bool
    from_diffusers_cache: bool
    path: os.PathLike
    original_filename: str

    constructor: Type[PipelineBase]
    init_dict: Dict[str, Tuple[str]] = {}

    scheduler: Any = None # for Img2Img

    def __init__(self, constructor: Type[PipelineBase], path: os.PathLike): # pylint: disable=super-init-not-called
        self._is_sdxl = check_pipeline_sdxl(constructor)
        self.from_diffusers_cache = check_diffusers_cache(path)
        self.path = path
        self.original_filename = os.path.basename(os.path.dirname(os.path.dirname(path)) if self.from_diffusers_cache else path)

        if os.path.isdir(path):
            self.init_dict = load_init_dict(constructor, path)
            self.scheduler = load_submodel(self.path, None, "scheduler", self.init_dict["scheduler"])
        else:
            cls = diffusers.StableDiffusionXLPipeline if self._is_sdxl else diffusers.StableDiffusionPipeline
            try:
                pipeline = cls.from_single_file(path)
                self.scheduler = pipeline.scheduler
                path = shared.opts.onnx_temp_dir
                if os.path.isdir(path):
                    shutil.rmtree(path)
                os.mkdir(path)
                pipeline.save_pretrained(path)
                del pipeline
                self.init_dict = load_init_dict(constructor, path)
            except Exception:
                log.error(f'ONNX: Failed to load ONNX pipeline: is_sdxl={self._is_sdxl}')
                log.warning('ONNX: You cannot load this model using the pipeline you selected. Please check Diffusers pipeline in Compute Settings.')
                return
        if "vae" in self.init_dict:
            del self.init_dict["vae"]

        self.is_refiner = self._is_sdxl and "Img2Img" not in constructor.__name__ and "Img2Img" in diffusers.DiffusionPipeline.load_config(path)["_class_name"]
        self.constructor = constructor
        if self.is_refiner:
            from modules.onnx_impl.pipelines.onnx_stable_diffusion_xl_img2img_pipeline import OnnxStableDiffusionXLImg2ImgPipeline
            self.constructor = OnnxStableDiffusionXLImg2ImgPipeline
        self.model_type = self.constructor.__name__

    def derive_properties(self, pipeline: diffusers.DiffusionPipeline):
        pipeline.sd_model_hash = self.sd_model_hash
        pipeline.sd_checkpoint_info = self.sd_checkpoint_info
        pipeline.sd_model_checkpoint = self.sd_model_checkpoint
        pipeline.scheduler = self.scheduler
        return pipeline

    def convert(self, submodels: List[str], in_dir: os.PathLike, out_dir: os.PathLike):
        shutil.rmtree("cache", ignore_errors=True)
        shutil.rmtree("footprints", ignore_errors=True)

        if shared.opts.onnx_cache_converted:
            shutil.copytree(
                in_dir, out_dir, ignore=shutil.ignore_patterns("weights.pb", "*.onnx", "*.safetensors", "*.ckpt")
            )

        from modules import olive_script as olv

        for submodel in submodels:
            destination = os.path.join(out_dir, submodel)

            if not os.path.isdir(destination):
                os.mkdir(destination)

            model = getattr(olv, f"{submodel}_load")(in_dir)
            sample = getattr(olv, f"{submodel}_conversion_inputs")(None)
            with tempfile.TemporaryDirectory(prefix="onnx_conversion") as temp_dir:
                temp_path = os.path.join(temp_dir, "model.onnx")
                torch.onnx.export(
                    model,
                    sample,
                    temp_path,
                    opset_version=14,
                    **get_io_config(submodel, self._is_sdxl),
                )
                model = onnx.load(temp_path)
            onnx.save_model(
                model,
                os.path.join(destination, "model.onnx"),
                save_as_external_data=submodel == "unet",
                all_tensors_to_one_file=True,
                location="weights.pb",
            )
            log.info(f"ONNX: Successfully exported converted model: submodel={submodel}")

        kwargs = {}

        init_dict = self.init_dict.copy()
        for submodel in submodels:
            kwargs[submodel] = diffusers.OnnxRuntimeModel.load_model(
                os.path.join(out_dir, submodel, "model.onnx"),
                provider=get_provider(),
            ) if self._is_sdxl else diffusers.OnnxRuntimeModel.from_pretrained(
                os.path.join(out_dir, submodel),
                provider=get_provider(),
            )
            if submodel in init_dict:
                del init_dict[submodel] # already loaded as OnnxRuntimeModel.
        kwargs.update(load_submodels(in_dir, self._is_sdxl, init_dict)) # load others.
        constructor = get_base_constructor(self.constructor, self.is_refiner)
        kwargs = patch_kwargs(constructor, kwargs)

        pipeline = constructor(**kwargs)
        model_index = json.loads(pipeline.to_json_string())
        del pipeline

        for k, v in init_dict.items(): # copy missing submodels. (ORTStableDiffusionXLPipeline)
            if k not in model_index:
                model_index[k] = v

        with open(os.path.join(out_dir, "model_index.json"), 'w', encoding="utf-8") as file:
            json.dump(model_index, file)

    def run_olive(self, submodels: List[str], in_dir: os.PathLike, out_dir: os.PathLike):
        if not shared.cmd_opts.debug:
            ort.set_default_logger_severity(4)

        try:
            from olive.model import ONNXModel # olive-ai==0.4.0
        except ImportError:
            from olive.model import ONNXModelHandler as ONNXModel # olive-ai==0.5.0

        shutil.rmtree("cache", ignore_errors=True)
        shutil.rmtree("footprints", ignore_errors=True)

        if shared.opts.olive_cache_optimized:
            shutil.copytree(
                in_dir, out_dir, ignore=shutil.ignore_patterns("weights.pb", "*.onnx", "*.safetensors", "*.ckpt")
            )

        optimized_model_paths = {}

        for submodel in submodels:
            log.info(f"\nProcessing {submodel}")

            with open(os.path.join(sd_configs_path, "olive", 'sdxl' if self._is_sdxl else 'sd', f"{submodel}.json"), "r", encoding="utf-8") as config_file:
                olive_config: Dict[str, Dict[str, Dict]] = json.load(config_file)

            for flow in olive_config["pass_flows"]:
                for i in range(len(flow)):
                    flow[i] = flow[i].replace("AutoExecutionProvider", shared.opts.onnx_execution_provider)
            olive_config["input_model"]["config"]["model_path"] = os.path.abspath(os.path.join(in_dir, submodel, "model.onnx"))
            olive_config["engine"]["execution_providers"] = [shared.opts.onnx_execution_provider]

            for pass_key in olive_config["passes"]:
                if olive_config["passes"][pass_key]["type"] == "OrtTransformersOptimization":
                    float16 = shared.opts.olive_float16 and not (submodel == "vae_encoder" and shared.opts.olive_vae_encoder_float32)
                    olive_config["passes"][pass_key]["config"]["float16"] = float16
                    if shared.opts.onnx_execution_provider == ExecutionProvider.CUDA or shared.opts.onnx_execution_provider == ExecutionProvider.ROCm:
                        if version.parse(ort.__version__) < version.parse("1.17.0"):
                            olive_config["passes"][pass_key]["config"]["optimization_options"] = {"enable_skip_group_norm": False}
                        if float16:
                            olive_config["passes"][pass_key]["config"]["keep_io_types"] = False

            run_olive_workflow(olive_config)

            with open(os.path.join("footprints", f"{submodel}_{EP_TO_NAME[shared.opts.onnx_execution_provider]}_footprints.json"), "r", encoding="utf-8") as footprint_file:
                footprints = json.load(footprint_file)
            processor_final_pass_footprint = None
            for _, footprint in footprints.items():
                if footprint["from_pass"] == olive_config["passes"][olive_config["pass_flows"][-1][-1]]["type"]:
                    processor_final_pass_footprint = footprint

            assert processor_final_pass_footprint, "Failed to optimize model"

            optimized_model_paths[submodel] = ONNXModel(
                **processor_final_pass_footprint["model_config"]["config"]
            ).model_path

            log.info(f"Olive: Successfully processed model: submodel={submodel}")

        for submodel in submodels:
            src_path = optimized_model_paths[submodel]
            src_parent = os.path.dirname(src_path)
            dst_parent = os.path.join(out_dir, submodel)
            dst_path = os.path.join(dst_parent, "model.onnx")
            if not os.path.isdir(dst_parent):
                os.mkdir(dst_parent)
            shutil.copyfile(src_path, dst_path)

            data_src_path = os.path.join(src_parent, (os.path.basename(src_path) + ".data"))
            if os.path.isfile(data_src_path):
                data_dst_path = os.path.join(dst_parent, (os.path.basename(dst_path) + ".data"))
                shutil.copyfile(data_src_path, data_dst_path)

            weights_src_path = os.path.join(src_parent, "weights.pb")
            if os.path.isfile(weights_src_path):
                weights_dst_path = os.path.join(dst_parent, "weights.pb")
                shutil.copyfile(weights_src_path, weights_dst_path)
        del optimized_model_paths

        kwargs = {}

        init_dict = self.init_dict.copy()
        for submodel in submodels:
            kwargs[submodel] = diffusers.OnnxRuntimeModel.load_model(
                os.path.join(out_dir, submodel, "model.onnx"),
                provider=get_provider(),
            ) if self._is_sdxl else diffusers.OnnxRuntimeModel.from_pretrained(
                os.path.join(out_dir, submodel),
                provider=get_provider(),
            )
            if submodel in init_dict:
                del init_dict[submodel] # already loaded as OnnxRuntimeModel.
        kwargs.update(load_submodels(in_dir, self._is_sdxl, init_dict)) # load others.
        constructor = get_base_constructor(self.constructor, self.is_refiner)
        kwargs = patch_kwargs(constructor, kwargs)

        pipeline = constructor(**kwargs)
        model_index = json.loads(pipeline.to_json_string())
        del pipeline

        for k, v in init_dict.items(): # copy missing submodels. (ORTStableDiffusionXLPipeline)
            if k not in model_index:
                model_index[k] = v

        with open(os.path.join(out_dir, "model_index.json"), 'w', encoding="utf-8") as file:
            json.dump(model_index, file)

    def preprocess(self, p: StableDiffusionProcessing):
        disable_classifier_free_guidance = p.cfg_scale < 0.01

        config.from_diffusers_cache = self.from_diffusers_cache
        config.is_sdxl = self._is_sdxl

        config.vae = os.path.join(models_path, "VAE", shared.opts.sd_vae)
        if not os.path.isfile(config.vae):
            del config.vae
        config.vae_sdxl_fp16_fix = self._is_sdxl and not shared.opts.diffusers_vae_upcast

        config.width = p.width
        config.height = p.height
        config.batch_size = p.batch_size

        if self._is_sdxl and not self.is_refiner:
            config.cross_attention_dim = 2048
            config.time_ids_size = 6
        else:
            config.cross_attention_dim = 768
            config.time_ids_size = 5

        if not disable_classifier_free_guidance and "turbo" in str(self.path).lower():
            log.warning("ONNX: It looks like you are trying to run a Turbo model with CFG Scale, which will lead to 'size mismatch' or 'unexpected parameter' error.")

        out_dir = os.path.join(shared.opts.onnx_cached_models_path, self.original_filename)
        if (self.from_diffusers_cache and check_cache_onnx(self.path)): # if model is ONNX format or had already converted, skip conversion.
            out_dir = self.path
        elif not os.path.isdir(out_dir):
            try:
                self.convert(
                    (SUBMODELS_SDXL_REFINER if self.is_refiner else SUBMODELS_SDXL) if self._is_sdxl else SUBMODELS_SD,
                    self.path if os.path.isdir(self.path) else shared.opts.onnx_temp_dir,
                    out_dir,
                )
            except Exception as e:
                log.error(f"ONNX: Failed to convert model: model='{self.original_filename}', error={e}")
                shutil.rmtree(shared.opts.onnx_temp_dir, ignore_errors=True)
                shutil.rmtree(out_dir, ignore_errors=True)
                return

        kwargs = {
            "provider": get_provider(),
        }
        in_dir = out_dir

        if shared.opts.cuda_compile_backend == "olive-ai":
            if run_olive_workflow is None:
                log.warning('Olive: Skipping model compilation because olive-ai was loaded unsuccessfully.')
            else:
                submodels_for_olive = []

                if "Text Encoder" in shared.opts.cuda_compile:
                    if not self.is_refiner:
                        submodels_for_olive.append("text_encoder")
                    if self._is_sdxl:
                        submodels_for_olive.append("text_encoder_2")
                if "Model" in shared.opts.cuda_compile:
                    submodels_for_olive.append("unet")
                if "VAE" in shared.opts.cuda_compile:
                    submodels_for_olive.append("vae_encoder")
                    submodels_for_olive.append("vae_decoder")

                if len(submodels_for_olive) == 0:
                    log.warning("Olive: Skipping olive run.")
                else:
                    log.warning("Olive implementation is experimental. It contains potentially an issue and is subject to change at any time.")

                    out_dir = os.path.join(shared.opts.onnx_cached_models_path, f"{self.original_filename}-{config.width}w-{config.height}h")
                    if not os.path.isdir(out_dir): # check the model is already optimized (cached)
                        if not shared.opts.olive_cache_optimized:
                            out_dir = shared.opts.onnx_temp_dir

                        if p.width != p.height:
                            log.warning("Olive: Different width and height are detected. The quality of the result is not guaranteed.")

                        if shared.opts.olive_static_dims:
                            sess_options = DynamicSessionOptions()
                            sess_options.enable_static_dims({
                                "is_sdxl": self._is_sdxl,
                                "is_refiner": self.is_refiner,

                                "hidden_batch_size": p.batch_size if disable_classifier_free_guidance else p.batch_size * 2,
                                "height": p.height,
                                "width": p.width,
                            })
                            kwargs["sess_options"] = sess_options

                        try:
                            self.run_olive(submodels_for_olive, in_dir, out_dir)
                        except Exception as e:
                            log.error(f"Olive: Failed to run olive passes: model='{self.original_filename}', error={e}")
                            shutil.rmtree(shared.opts.onnx_temp_dir, ignore_errors=True)
                            shutil.rmtree(out_dir, ignore_errors=True)

        pipeline = self.derive_properties(load_pipeline(self.constructor, out_dir, **kwargs))

        if not shared.opts.onnx_cache_converted and in_dir != self.path:
            shutil.rmtree(in_dir)
        shutil.rmtree(shared.opts.onnx_temp_dir, ignore_errors=True)

        return pipeline
