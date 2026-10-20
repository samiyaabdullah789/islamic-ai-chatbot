
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class SourceStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceCreate(BaseModel):
    name: str
    file_path: str


class SourceStatusUpdate(BaseModel):
    status: SourceStatus


class SourceResponse(BaseModel):
    id: int
    name: str
    file_path: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
