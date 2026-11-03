
import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_PATH = BASE_DIR / "chroma_data"


class ChromaClient:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        # Existing collection: preserve all old RAG data
        self.collection = self.client.get_or_create_collection(
            name="islamic_knowledge"
        )

        # New collection: question-based Q&A retrieval
        self.qa_collection = self.client.get_or_create_collection(
            name="islamic_qa_questions"
        )


chroma_client = ChromaClient()
