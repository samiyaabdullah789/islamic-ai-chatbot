from sqlalchemy.orm import Session

from app.models.source import Source
from app.repositories.source_repository import source_repository


class SourceService:

    def register_source(
        self,
        db: Session,
        name: str,
        file_path: str,
    ) -> Source:

        if not name.strip():
            raise ValueError("Source name is required")

        if not file_path.strip():
            raise ValueError("File path is required")

        return source_repository.create(
            db=db,
            name=name.strip(),
            file_path=file_path.strip(),
        )

    def get_source(
        self,
        db: Session,
        source_id: int,
    ) -> Source | None:

        return source_repository.get_by_id(
            db=db,
            source_id=source_id,
        )


source_service = SourceService()