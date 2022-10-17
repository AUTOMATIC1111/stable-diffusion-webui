from inflection import underscore
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, create_model
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
import inspect


class ModelDef(BaseModel):
    """Assistance Class for Pydantic Dynamic Model Generation"""

    field: str
    field_alias: str
    field_type: Any
    field_value: Any


class pydanticModelGenerator:
    """
    Takes in created classes and stubs them out in a way FastAPI/Pydantic is happy about:
    source_data is a snapshot of the default values produced by the class
    params are the names of the actual keys required by __init__
    """

    def __init__(
        self,
        model_name: str = None,
        source_data: {} = {},
        params: Dict = {},
        overrides: Dict = {},
        optionals: Dict = {},
    ):
        def field_type_generator(k, v, overrides, optionals):
            print(k, v)
            field_type = str if not overrides.get(k) else overrides[k]["type"]
            if v is None:
                field_type = Any
            else:
                field_type = type(v)
            
            return Optional[field_type]
        
        self._model_name = model_name
        self._json_data = source_data
        self._model_def = [
            ModelDef(
                field=underscore(k),
                field_alias=k,
                field_type=field_type_generator(k, v, overrides, optionals),
                field_value=v
            )
            for (k,v) in source_data.items() if k in params
        ]

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
        return DynamicModel
    
StableDiffusionProcessingAPI = pydanticModelGenerator("StableDiffusionProcessing", 
                                                      StableDiffusionProcessing().__dict__, 
                                                      inspect.signature(StableDiffusionProcessing.__init__).parameters).generate_model()
