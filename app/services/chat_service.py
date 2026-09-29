class ChatService:

    async def process_message(self, message: str) -> str:
        return f"You said: {message}"


chat_service = ChatService()