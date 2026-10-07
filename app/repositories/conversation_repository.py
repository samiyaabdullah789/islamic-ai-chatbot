from sqlalchemy.orm import Session

from app.models.conversation import Conversation


class ConversationRepository:

    def create(
        self,
        db: Session,
        user_id: str,
        title: str | None = None,
    ) -> Conversation:

        conversation = Conversation(
            user_id=user_id,
            title=title,
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return conversation

    def get_by_id_and_user(
        self,
        db: Session,
        conversation_id: int,
        user_id: str,
    ) -> Conversation | None:

        return (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
            .first()
        )

    def get_all_by_user(
        self,
        db: Session,
        user_id: str,
    ) -> list[Conversation]:

        return (
            db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
            .all()
        )


conversation_repository = ConversationRepository()