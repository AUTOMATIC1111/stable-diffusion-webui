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
    Takes source_data:Dict ( a single instance example of something like a JSON node) and self generates a pythonic data model with Alias to original source field names. This makes it easy to popuate or export to other systems yet handle the data in a pythonic way.
    Being a pydantic datamodel all the richness of pydantic data validation is available and these models can easily be used in FastAPI and or a ORM

    It does not process full JSON data structures but takes simple JSON document with basic elements

    Provide a model_name, an example of JSON data and a dict of type overrides

    Example:

    source_data = {'Name': '48 Rainbow Rd',
        'GroupAddressStyle': 'ThreeLevel',
        'LastModified': '2020-12-21T07:02:51.2400232Z',
        'ProjectStart': '2020-12-03T07:36:03.324856Z',
        'Comment': '',
        'CompletionStatus': 'Editing',
        'LastUsedPuid': '955',
        'Guid': '0c85957b-c2ae-4985-9752-b300ab385b36'}

    source_overrides = {'Guid':{'type':uuid.UUID},
            'LastModified':{'type':datetime },
            'ProjectStart':{'type':datetime },
            }
    source_optionals = {"Comment":True}

    #create Model
    model_Project=pydanticModelGenerator(
        model_name="Project",
        source_data=source_data,
        overrides=source_overrides,
        optionals=source_optionals).generate_model()

    #create instance using DynamicModel
    project_instance=model_Project(**project_info)

    """

    def __init__(
        self,
        model_name: str = None,
        source_data: str = None,
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