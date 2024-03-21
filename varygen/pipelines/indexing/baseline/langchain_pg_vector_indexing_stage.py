import logging
from typing import Optional

from varygen.common.utils import get_pg_vector_from_config
from varygen.config import ExperimentPipelineConfig
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.pgvector import PGVector
from surround import Stage

from ..state import IndexingState

LOG = logging.getLogger(__name__)


class LangchainPGVectorIndexingStage(Stage):
    """
    Vectorises the final chunks and inserts them into the PG Vector store.
    """

    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
        self.pg_vector: Optional[PGVector] = None

    def initialise(self, config: ExperimentPipelineConfig):
        """
        Initialise the stage with the configuration.
        """

        self.pg_vector = get_pg_vector_from_config(self.embedding_model, config)

    def operate(self, state: IndexingState, config: ExperimentPipelineConfig):  # pylint: disable=unused-argument
        """
        Vectorise the final chunks and add them to the PG Vector store.
        """

        if not state.final_chunks:
            LOG.warning("No chunks to index")
            return

        if not self.pg_vector:
            LOG.error("Stage has not been initialised")
            raise RuntimeError("Stage has not been initialised")

        texts = [chunk.page_content for chunk in state.final_chunks]
        metadatas = [chunk.metadata for chunk in state.final_chunks]

        embeddings = self.embedding_model.embed_documents(texts)
        for chunk, embedding in zip(state.final_chunks, embeddings):
            # Track the embeddings created
            chunk.embedding = embedding
            if hasattr(self.embedding_model, "model"):
                chunk.metadata["embedding_model"] = self.embedding_model.model

        if self.pg_vector:
            self.pg_vector.add_embeddings(texts, embeddings, metadatas)
