import json
from typing import List, Optional
from uuid import UUID, uuid4

from langchain.schema.document import Document as LangchainDocument
from llama_index.schema import Document as LlamaIndexDocument


class Document:
    """
    Library agnostic document class.
    """

    id: UUID
    page_content: str
    metadata: dict
    embedding: Optional[List[float]]
    score: Optional[float]

    def __init__(self, page_content, metadata):
        self.id = uuid4()  # pylint: disable=C0103
        self.page_content = page_content
        self.metadata = metadata
        self.embedding = None
        self.score = None

    def set_score(self, score) -> "Document":
        """
        Set the retrieval score for the document.
        """
        self.score = score
        return self

    def to_langchain_document(self) -> LangchainDocument:
        """
        Convert the document to a Langchain compatible document.
        """
        metadata = self.metadata.copy()
        metadata["id"] = str(self.id)
        return LangchainDocument(
            page_content=self.page_content,
            metadata=metadata,
        )

    def to_llama_index_document(self) -> LlamaIndexDocument:
        """
        Convert the document to a Llama Index compatible document.
        """
        return LlamaIndexDocument(
            text=self.page_content,
            metadata=self.metadata,
        )

    def to_json(self) -> str:
        """
        Convert the document to a JSON string.
        """
        return json.dumps(
            {
                "id": str(self.id),
                "page_content": self.page_content,
                "metadata": self.metadata,
                "embedding": self.embedding,
                "score": self.score,
            }
        )

    @classmethod
    def from_langchain_document(cls, document: LangchainDocument, score: Optional[float] = None):
        """
        Create a document from a Langchain compatible document.
        """
        return cls(
            page_content=document.page_content,
            metadata=document.metadata,
        ).set_score(score)

    @classmethod
    def from_llama_index_document(cls, document: LlamaIndexDocument, score: Optional[float] = None):
        """
        Create a document from a Llama Index compatible document.
        """
        return cls(
            page_content=document.text,
            metadata=document.metadata,
        ).set_score(score)
