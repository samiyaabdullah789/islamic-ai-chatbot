
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

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
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

        return len(document_ids)


chroma_store = ChromaStore()
