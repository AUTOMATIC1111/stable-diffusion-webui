from pydantic import BaseModel, Field, Json

class TextToImageResponse(BaseModel):
    images: list[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: Json
    info: Json

    