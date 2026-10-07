from sqlalchemy.orm import Session

from app.rag.pipeline import rag_pipeline
from app.repositories.conversation_repository import conversation_repository
from app.repositories.message_repository import message_repository


class ChatService:

    async def process_message(
        self,
        db: Session,
        user_id: str,
        message: str,
        conversation_id: int,
    ) -> str:

        # Check that conversation exists and belongs to current user
        conversation = conversation_repository.get_by_id_and_user(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if conversation is None:
            raise ValueError("Conversation not found")

        # Get previous conversation history
        previous_messages = message_repository.get_all_by_conversation(
            db=db,
            conversation_id=conversation_id,
        )

        # Convert previous database messages into LLM message format
        conversation_history = [
            {
                "role": previous_message.role,
                "content": previous_message.content,
            }
            for previous_message in previous_messages
        ]

        # Save current user message
        message_repository.create(
            db=db,
            conversation_id=conversation.id,
            role="user",
            content=message,
        )

        # Generate answer using RAG + conversation context
        response = await rag_pipeline.generate(
            question=message,
            conversation_history=conversation_history,
        )

        # Save assistant response
        message_repository.create(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            content=response,
        )

        return response


chat_service = ChatService()