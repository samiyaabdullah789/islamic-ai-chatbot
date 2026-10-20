
from datetime import datetime

from pydantic import BaseModel, Field


class SavedAnswerCreate(BaseModel):
    message_id: int = Field(gt=0)


class SavedAnswerResponse(BaseModel):
    id: int
    user_id: str
    message_id: int
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class SavedAnswerDetailResponse(SavedAnswerResponse):
    answer: str
