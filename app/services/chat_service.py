
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
        madhhab: str | None = None,
    ) -> dict:

        total_start = time.perf_counter()

        # -----------------------------------
        # 1. Verify conversation
        # -----------------------------------
        start = time.perf_counter()

        conversation = conversation_repository.get_by_id_and_user(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        print(
            f"[TIMING] Conversation check: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if conversation is None:
            raise ValueError("Conversation not found")

        # -----------------------------------
        # 2. Fetch previous messages
        # -----------------------------------
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
            f"[TIMING] Conversation history: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # -----------------------------------
        # 3. Save user message
        # -----------------------------------
        start = time.perf_counter()

        message_repository.create(
            db=db,
            conversation_id=conversation.id,
            role="user",
            content=message,
        )

        print(
            f"[TIMING] Save user message: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # -----------------------------------
        # 4. Generate RAG answer
        # -----------------------------------
        start = time.perf_counter()

        print(
            f"[CHAT] Selected Madhhab: "
            f"{madhhab or 'General'}",
            flush=True,
        )

        response = await rag_pipeline.generate(
            question=message,
            conversation_history=conversation_history,
            madhhab=madhhab,
        )

        print(
            f"[TIMING] RAG + LLM total: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # -----------------------------------
        # 5. Save assistant response
        # -----------------------------------
        start = time.perf_counter()

        assistant_message = message_repository.create(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            content=response,
        )

        print(
            f"[TIMING] Save assistant message: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # -----------------------------------
        # 6. Request timing
        # -----------------------------------
        print(
            f"[TIMING] COMPLETE CHAT REQUEST: "
            f"{time.perf_counter() - total_start:.2f}s",
            flush=True,
        )

        return {
            "response": response,
            "message_id": assistant_message.id,
        }


chat_service = ChatService()
