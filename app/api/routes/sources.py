
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.source import (
    SourceCreate,
    SourceResponse,
    SourceStatusUpdate,
)
from app.services.source_service import source_service


router = APIRouter(
    prefix="/sources",
    tags=["Sources"],
)


# -----------------------------------
# POST - Register a New Source
# -----------------------------------

@router.post(
    "/",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_source(
    payload: SourceCreate,
    db: Session = Depends(get_db),
):
    return source_service.register_source(
        db=db,
        name=payload.name,
        file_path=payload.file_path,
    )


# -----------------------------------
# GET - All Registered Sources
# -----------------------------------

@router.get(
    "/",
    response_model=list[SourceResponse],
)
def get_all_sources(
    db: Session = Depends(get_db),
):
    return source_service.get_all_sources(db=db)


# -----------------------------------
# GET - Safe Delete Preview
# -----------------------------------

@router.get("/{source_id}/delete-preview")
def preview_source_deletion(
    source_id: int,
    db: Session = Depends(get_db),
):
    source = source_service.get_source(
        db=db,
        source_id=source_id,
    )

    if source is None:
        raise HTTPException(
            status_code=404,
            detail="Source not found",
        )

    return source_service.delete_source(
        db=db,
        source_id=source_id,
        confirm=False,
    )


# -----------------------------------
# PATCH - Update Source Status
# -----------------------------------

@router.patch(
    "/{source_id}/status",
    response_model=SourceResponse,
)
def update_source_status(
    source_id: int,
    payload: SourceStatusUpdate,
    db: Session = Depends(get_db),
):
    source = source_service.update_source_status(
        db=db,
        source_id=source_id,
        new_status=payload.status.value,
    )

    if source is None:
        raise HTTPException(
            status_code=404,
            detail="Source not found",
        )

    return source


# -----------------------------------
# GET - Source by ID
# -----------------------------------

@router.get(
    "/{source_id}",
    response_model=SourceResponse,
)
def get_source_by_id(
    source_id: int,
    db: Session = Depends(get_db),
):
    source = source_service.get_source(
        db=db,
        source_id=source_id,
    )

    if source is None:
        raise HTTPException(
            status_code=404,
            detail="Source not found",
        )

    return source
