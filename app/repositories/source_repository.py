
from sqlalchemy.orm import Session

from app.models.source import Source


class SourceRepository:

    def create(
        self,
        db: Session,
        name: str,
        file_path: str,
    ) -> Source:

        source = Source(
            name=name,
            file_path=file_path,
            status="pending",
        )

        db.add(source)
        db.commit()
        db.refresh(source)

        return source

    def get_by_id(
        self,
        db: Session,
        source_id: int,
    ) -> Source | None:

        return db.get(Source, source_id)

    def get_all(
        self,
        db: Session,
    ) -> list[Source]:

        return (
            db.query(Source)
            .order_by(Source.id.desc())
            .all()
        )

    def update_status(
        self,
        db: Session,
        source: Source,
        new_status: str,
    ) -> Source:

        source.status = new_status

        db.commit()
        db.refresh(source)

        return source

    def delete(
        self,
        db: Session,
        source: Source,
    ) -> None:

        db.delete(source)
        db.commit()


source_repository = SourceRepository()
