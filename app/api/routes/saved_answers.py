
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.saved_answer_repository import (
    saved_answer_repository,
)
from app.schemas.saved_answer import (
    SavedAnswerCreate,
    SavedAnswerResponse,
    SavedAnswerDetailResponse,
)


router = APIRouter(prefix="/saved-answers")

# Development only.
# Replace with authenticated user ID later.
DEV_USER_ID = "1"


# -----------------------------------
# POST - Save an AI Answer
# -----------------------------------

@router.post(
    "",
    response_model=SavedAnswerResponse,
    status_code=status.HTTP_201_CREATED,
)
def save_answer(
    payload: SavedAnswerCreate,
    db: Session = Depends(get_db),
):
    user_id = DEV_USER_ID

    message = saved_answer_repository.get_message_for_user(
        db=db,
        message_id=payload.message_id,
        user_id=user_id,
    )

    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AI message not found",
        )

    existing = saved_answer_repository.get_existing(
        db=db,
        user_id=user_id,
        message_id=payload.message_id,
    )

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Answer already saved",
        )

    return saved_answer_repository.create(
        db=db,
        user_id=user_id,
        message_id=payload.message_id,
    )


# -----------------------------------
# GET - All Saved Answers With Text
# -----------------------------------

@router.get(
    "",
    response_model=list[SavedAnswerDetailResponse],
)
def get_saved_answers(
    db: Session = Depends(get_db),
):
    user_id = DEV_USER_ID

    results = saved_answer_repository.get_all_with_answers(
        db=db,
        user_id=user_id,
    )

    saved_answers = []

    for saved_answer, answer_text in results:
        saved_answers.append(
            SavedAnswerDetailResponse(
                id=saved_answer.id,
                user_id=saved_answer.user_id,
                message_id=saved_answer.message_id,
                created_at=saved_answer.created_at,
                answer=answer_text,
            )
        )

    return saved_answers


# -----------------------------------
# DELETE - Unsave an Answer
# -----------------------------------

@router.delete(
    "/{saved_answer_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_saved_answer(
    saved_answer_id: int,
    db: Session = Depends(get_db),
):
    saved_answer = saved_answer_repository.get_by_id_and_user(
        db=db,
        saved_answer_id=saved_answer_id,
        user_id=DEV_USER_ID,
    )

    if saved_answer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved answer not found",
        )

    saved_answer_repository.delete(
        db=db,
        saved_answer=saved_answer,
    )

    return None
