
from sqlalchemy.orm import Session

from app.models.saved_answer import SavedAnswer
from app.models.message import Message
from app.models.conversation import Conversation


class SavedAnswerRepository:

    def get_message_for_user(
        self,
        db: Session,
        message_id: int,
        user_id: str,
    ) -> Message | None:

        return (
            db.query(Message)
            .join(
                Conversation,
                Message.conversation_id == Conversation.id,
            )
            .filter(
                Message.id == message_id,
                Message.role == "assistant",
                Conversation.user_id == user_id,
            )
            .first()
        )

    def get_existing(
        self,
        db: Session,
        user_id: str,
        message_id: int,
    ) -> SavedAnswer | None:

        return (
            db.query(SavedAnswer)
            .filter(
                SavedAnswer.user_id == user_id,
                SavedAnswer.message_id == message_id,
            )
            .first()
        )

    def create(
        self,
        db: Session,
        user_id: str,
        message_id: int,
    ) -> SavedAnswer:

        saved_answer = SavedAnswer(
            user_id=user_id,
            message_id=message_id,
        )

        db.add(saved_answer)
        db.commit()
        db.refresh(saved_answer)

        return saved_answer

    def get_all_by_user(
        self,
        db: Session,
        user_id: str,
    ) -> list[SavedAnswer]:

        return (
            db.query(SavedAnswer)
            .filter(SavedAnswer.user_id == user_id)
            .order_by(
                SavedAnswer.created_at.desc(),
                SavedAnswer.id.desc(),
            )
            .all()
        )

    def get_all_with_answers(
        self,
        db: Session,
        user_id: str,
    ) -> list[tuple[SavedAnswer, str]]:

        return (
            db.query(
                SavedAnswer,
                Message.content,
            )
            .join(
                Message,
                SavedAnswer.message_id == Message.id,
            )
            .join(
                Conversation,
                Message.conversation_id == Conversation.id,
            )
            .filter(
                SavedAnswer.user_id == user_id,
                Conversation.user_id == user_id,
                Message.role == "assistant",
            )
            .order_by(
                SavedAnswer.created_at.desc(),
                SavedAnswer.id.desc(),
            )
            .all()
        )

    def get_by_id_and_user(
        self,
        db: Session,
        saved_answer_id: int,
        user_id: str,
    ) -> SavedAnswer | None:

        return (
            db.query(SavedAnswer)
            .filter(
                SavedAnswer.id == saved_answer_id,
                SavedAnswer.user_id == user_id,
            )
            .first()
        )

    def delete(
        self,
        db: Session,
        saved_answer: SavedAnswer,
    ) -> None:

        db.delete(saved_answer)
        db.commit()


saved_answer_repository = SavedAnswerRepository()
