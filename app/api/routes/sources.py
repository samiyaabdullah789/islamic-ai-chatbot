from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.source import SourceCreate, SourceResponse
from app.services.source_service import source_service

router = APIRouter(prefix="/sources", tags=["Sources"])


@router.post("/", response_model=SourceResponse, status_code=201)
def register_source(
    payload: SourceCreate,
    db: Session = Depends(get_db),
):
    return source_service.register_source(
        db=db,
        name=payload.name,
        file_path=payload.file_path,
    )