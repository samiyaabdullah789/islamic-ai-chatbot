
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ConversationCreate(BaseModel):
    title: str | None = None


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Conversation title cannot be empty")

        return value


class ConversationResponse(BaseModel):
    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }
