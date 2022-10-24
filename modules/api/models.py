from array import array
from inflection import underscore
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, create_model
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
import inspect


API_NOT_ALLOWED = [
    "self",
    "kwargs",
    "sd_model",
    "outpath_samples",
    "outpath_grids",
    "sampler_index",
    "do_not_save_samples",
    "do_not_save_grid",
    "extra_generation_params",
    "overlay_images",
    "do_not_reload_embeddings",
    "seed_enable_extras",
    "prompt_for_display",
    "sampler_noise_scheduler_override",
    "ddim_discretize"
]

class ModelDef(BaseModel):
    """Assistance Class for Pydantic Dynamic Model Generation"""

    field: str
    field_alias: str
    field_type: Any
    field_value: Any


class PydanticModelGenerator:
    """
    Takes in created classes and stubs them out in a way FastAPI/Pydantic is happy about:
    source_data is a snapshot of the default values produced by the class
    params are the names of the actual keys required by __init__
    """

    def __init__(
        self,
        model_name: str = None,
        class_instance = None,
        additional_fields = None,
    ):
        def field_type_generator(k, v):
            # field_type = str if not overrides.get(k) else overrides[k]["type"]
            # print(k, v.annotation, v.default)
            field_type = v.annotation
            
            return Optional[field_type]
        
        def merge_class_params(class_):
            all_classes = list(filter(lambda x: x is not object, inspect.getmro(class_)))
            parameters = {}
            for classes in all_classes:
                parameters = {**parameters, **inspect.signature(classes.__init__).parameters}
            return parameters
            
                
        self._model_name = model_name
        self._class_data = merge_class_params(class_instance)
        self._model_def = [
            ModelDef(
                field=underscore(k),
                field_alias=k,
                field_type=field_type_generator(k, v),
                field_value=v.default
            )
            for (k,v) in self._class_data.items() if k not in API_NOT_ALLOWED
        ]
        
        for fields in additional_fields:
            self._model_def.append(ModelDef(
                field=underscore(fields["key"]), 
                field_alias=fields["key"], 
                field_type=fields["type"],
                field_value=fields["default"]))

    def generate_model(self):
        """
        Creates a pydantic BaseModel
        from the json and overrides provided at initialization
        """
        fields = {
            d.field: (d.field_type, Field(default=d.field_value, alias=d.field_alias)) for d in self._model_def
        }
        DynamicModel = create_model(self._model_name, **fields)
        DynamicModel.__config__.allow_population_by_field_name = True
        DynamicModel.__config__.allow_mutation = True
        return DynamicModel
    
StableDiffusionTxt2ImgProcessingAPI = PydanticModelGenerator(
    "StableDiffusionProcessingTxt2Img", 
    StableDiffusionProcessingTxt2Img,
    [{"key": "sampler_index", "type": str, "default": "Euler"}]
).generate_model()

StableDiffusionImg2ImgProcessingAPI = PydanticModelGenerator(
    "StableDiffusionProcessingImg2Img", 
    StableDiffusionProcessingImg2Img,
    [{"key": "sampler_index", "type": str, "default": "Euler"}, {"key": "init_images", "type": list, "default": None}, {"key": "denoising_strength", "type": float, "default": 0.75}, {"key": "mask", "type": str, "default": None}]
).generate_model()