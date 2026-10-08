
import time

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

        total_start = time.perf_counter()

        # Check that conversation exists and belongs to current user
        start = time.perf_counter()

        conversation = conversation_repository.get_by_id_and_user(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        print(
            f"[TIMING] Conversation check: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if conversation is None:
            raise ValueError("Conversation not found")

        # Get previous conversation history
        start = time.perf_counter()

        previous_messages = message_repository.get_all_by_conversation(
            db=db,
            conversation_id=conversation_id,
        )

        conversation_history = [
            {
                "role": previous_message.role,
                "content": previous_message.content,
            }
            for previous_message in previous_messages
        ]

        print(
            f"[TIMING] Conversation history: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # Save current user message
        start = time.perf_counter()

        message_repository.create(
            db=db,
            conversation_id=conversation.id,
            role="user",
            content=message,
        )

        print(
            f"[TIMING] Save user message: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # Generate answer using RAG + conversation context
        start = time.perf_counter()

        response = await rag_pipeline.generate(
            question=message,
            conversation_history=conversation_history,
        )

        print(
            f"[TIMING] RAG + LLM total: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # Save assistant response
        start = time.perf_counter()

        message_repository.create(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            content=response,
        )

        print(
            f"[TIMING] Save assistant message: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        print(
            f"[TIMING] COMPLETE CHAT REQUEST: {time.perf_counter() - total_start:.2f}s",
            flush=True,
        )

        return response


chat_service = ChatService()
