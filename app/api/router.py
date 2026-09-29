from fastapi import APIRouter

from app.api.routes import health, chat


api_router = APIRouter()

api_router.include_router(
    health.router,
    tags=["Health"],
)

api_router.include_router(
    chat.router,
    tags=["Chat"],
)