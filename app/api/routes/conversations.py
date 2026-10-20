
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

# Temporary JWT authentication - disabled for now
# from app.api.dependencies import get_current_user_id

from app.db.session import get_db
from app.repositories.conversation_repository import conversation_repository
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
)
from app.repositories.message_repository import message_repository
from app.schemas.message import MessageResponse


router = APIRouter(prefix="/conversations")

# Temporary development user.
# Must be replaced with real app authentication before public deployment.
DEV_USER_ID = "1"


@router.post("", response_model=ConversationResponse)
def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db),
):
    user_id = DEV_USER_ID

    conversation = conversation_repository.create(
        db=db,
        user_id=user_id,
        title=request.title,
    )

    return conversation


@router.get("", response_model=list[ConversationResponse])
def get_conversations(
    db: Session = Depends(get_db),
):
    user_id = DEV_USER_ID

    conversations = conversation_repository.get_all_by_user(
        db=db,
        user_id=user_id,
    )

    return conversations


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
):
    user_id = DEV_USER_ID

    conversation = conversation_repository.get_by_id_and_user(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return conversation


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def rename_conversation(
    conversation_id: int,
    request: ConversationUpdate,
    db: Session = Depends(get_db),
):
    user_id = DEV_USER_ID

    conversation = conversation_repository.get_by_id_and_user(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    updated_conversation = conversation_repository.update_title(
        db=db,
        conversation=conversation,
        title=request.title,
    )

    return updated_conversation


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
):
    user_id = DEV_USER_ID

    conversation = conversation_repository.get_by_id_and_user(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages = message_repository.get_all_by_conversation(
        db=db,
        conversation_id=conversation_id,
    )

    return messages


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
):
    user_id = DEV_USER_ID

    conversation = conversation_repository.get_by_id_and_user(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    conversation_repository.delete(
        db=db,
        conversation=conversation,
    )

    return None
