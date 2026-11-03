
from typing import Literal

from pydantic import BaseModel, field_validator


Madhhab = Literal[
    "Hanafi",
    "Shafi'i",
    "Maliki",
    "Hanbali",
]


class ChatRequest(BaseModel):
    conversation_id: int
    message: str

    # General mode is represented internally by None
    madhhab: Madhhab | None = None

    @field_validator("madhhab", mode="before")
    @classmethod
    def normalize_madhhab(cls, value):

        # Null means General
        if value is None:
            return None

        if not isinstance(value, str):
            return value

        value = value.strip()

        # Empty string means General
        if not value:
            return None

        # General means no specific Madhhab
        if value.casefold() == "general":
            return None

        names = {
            "hanafi": "Hanafi",
            "shafi": "Shafi'i",
            "shafi'i": "Shafi'i",
            "shafii": "Shafi'i",
            "maliki": "Maliki",
            "hanbali": "Hanbali",
        }

        return names.get(
            value.casefold(),
            value,
        )


class ChatResponse(BaseModel):
    response: str
    message_id: int
