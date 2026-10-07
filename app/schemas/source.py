from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SourceCreate(BaseModel):
    name: str
    file_path: str


class SourceResponse(BaseModel):
    id: int
    name: str
    file_path: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)