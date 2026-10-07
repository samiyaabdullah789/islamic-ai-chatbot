from fastapi import APIRouter

from app.api.routes import health, chat, conversations, sources

api_router = APIRouter()

api_router.include_router(
    health.router,
    tags=["Health"],
)

api_router.include_router(
    chat.router,
    tags=["Chat"],
)

api_router.include_router(
    conversations.router,
    tags=["Conversations"],
)

api_router.include_router(
    sources.router,
)