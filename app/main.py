from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.router import api_router
from app.core.config import settings


STATIC_DIR = Path(__file__).parent / "static"


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

app.include_router(
    api_router,
    prefix=settings.API_PREFIX,
)


# Local testing UI for the chatbot
@app.get("/", include_in_schema=False)
def chat_ui():
    return FileResponse(STATIC_DIR / "chat.html")