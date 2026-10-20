
import logging

from sqlalchemy.orm import Session

from app.models.source import Source
from app.repositories.source_repository import source_repository
from app.vector_store.chroma_store import chroma_store


logger = logging.getLogger(__name__)


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

    def get_all_sources(
        self,
        db: Session,
    ) -> list[Source]:

        return source_repository.get_all(
            db=db,
        )

    def update_source_status(
        self,
        db: Session,
        source_id: int,
        new_status: str,
    ) -> Source | None:

        valid_statuses = {
            "pending",
            "processing",
            "completed",
            "failed",
        }

        if new_status not in valid_statuses:
            raise ValueError(
                "Invalid source status"
            )

        source = source_repository.get_by_id(
            db=db,
            source_id=source_id,
        )

        if source is None:
            return None

        return source_repository.update_status(
            db=db,
            source=source,
            new_status=new_status,
        )

    def get_source_document_count(
        self,
        source_id: int,
    ) -> int:

        document_ids = chroma_store.get_source_document_ids(
            source_id=source_id,
        )

        return len(document_ids)

    def delete_source(
        self,
        db: Session,
        source_id: int,
        confirm: bool = False,
    ) -> dict:

        source = source_repository.get_by_id(
            db=db,
            source_id=source_id,
        )

        if source is None:
            raise ValueError("Source not found")

        document_count = self.get_source_document_count(
            source_id=source_id,
        )

        if not confirm:
            return {
                "source_id": source_id,
                "source_name": source.name,
                "chroma_documents": document_count,
                "deleted": False,
                "confirmation_required": True,
            }

        deleted_count = chroma_store.delete_source_documents(
            source_id=source_id,
        )

        remaining_count = self.get_source_document_count(
            source_id=source_id,
        )

        if remaining_count != 0:
            raise RuntimeError(
                "ChromaDB deletion could not be verified. "
                "PostgreSQL source was not deleted."
            )

        try:
            source_repository.delete(
                db=db,
                source=source,
            )
        except Exception:
            db.rollback()
            logger.exception(
                "PostgreSQL deletion failed for source %s "
                "after ChromaDB deletion",
                source_id,
            )
            raise

        return {
            "source_id": source_id,
            "source_name": source.name,
            "chroma_documents_deleted": deleted_count,
            "deleted": True,
        }


source_service = SourceService()
