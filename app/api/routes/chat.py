from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

# Temporary JWT authentication - disabled for now
# from app.api.dependencies import get_current_user_id

from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,

    # Temporary JWT authentication - disabled for now
    # user_id: str = Depends(get_current_user_id),

    db: Session = Depends(get_db),
):

    # Temporary development user until app authentication is integrated
    user_id = "1"

    response = await chat_service.process_message(
        db=db,
        user_id=user_id,
        message=request.message,
        conversation_id=request.conversation_id,
    )

    return ChatResponse(
        response=response
    )