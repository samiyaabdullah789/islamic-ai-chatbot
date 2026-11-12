
from app.vector_store.chroma_client import chroma_client


class ChromaStore:

    def __init__(self):
        self.collection = chroma_client.collection

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ):
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Refresh BM25 on the next search.
        # Lazy import prevents circular imports.
        from app.rag.bm25_index import bm25_index
        bm25_index.invalidate()

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        where: dict | None = None,
    ):
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where is not None:
            query_args["where"] = where

        return self.collection.query(**query_args)

    def get_documents_page(
        self,
        offset: int = 0,
        limit: int = 500,
    ):
        """Read existing documents in batches."""
        return self.collection.get(
            offset=offset,
            limit=limit,
            include=[
                "documents",
                "metadatas",
            ],
        )

    def get_source_document_ids(
        self,
        source_id: int,
    ) -> list[str]:
        results = self.collection.get(
            where={"source_id": source_id},
            include=[],
        )
        return results["ids"]

    def delete_source_documents(
        self,
        source_id: int,
    ) -> int:
        document_ids = self.get_source_document_ids(
            source_id=source_id,
        )

        if not document_ids:
            return 0

        self.collection.delete(
            ids=document_ids,
        )

        from app.rag.bm25_index import bm25_index
        bm25_index.invalidate()

        return len(document_ids)


chroma_store = ChromaStore()
