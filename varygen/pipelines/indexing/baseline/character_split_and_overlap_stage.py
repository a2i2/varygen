import logging
from typing import Optional

from varygen.common.document import Document
from varygen.config import ExperimentPipelineConfig
from surround import Stage

from ..state import IndexingState

LOG = logging.getLogger(__name__)


class CharacterSplitAndOverlapStage(Stage):
    """
    Splits the text into chunks of a given size.
    """

    def __init__(self, experiment_name: str):
        super().__init__()
        self.experiment_name = experiment_name
        self.chunk_size: Optional[int] = None
        self.overlap_size: Optional[int] = None

    def initialise(self, config: ExperimentPipelineConfig):
        """
        Fetch the chunk size and overlap size from the config.
        """
        self.chunk_size = config.indexing[self.experiment_name].max_chunk_characters
        self.overlap_size = config.indexing[self.experiment_name].overlap_characters

    def operate(self, state: IndexingState, config: ExperimentPipelineConfig):  # pylint: disable=unused-argument
        """
        Split the text into chunks of a given max character size and overlap size.
        """

        if not self.chunk_size or not self.overlap_size:
            LOG.error("Stage has not been initialised")
            raise RuntimeError("Stage has not been initialised")

        new_chunks = []
        for chunk in state.loaded_chunks:
            chunk_index = 0

            if len(chunk.page_content) > self.chunk_size:
                for i in range(0, len(chunk.page_content), self.chunk_size):
                    new_content = chunk.page_content[max(0, i - self.overlap_size) : i + self.chunk_size]
                    new_metadata = chunk.metadata.copy()

                    new_chunk = Document(new_content, new_metadata)
                    new_chunk.metadata["chunk_index"] = chunk_index
                    new_chunks.append(new_chunk)

                    chunk_index += 1
            else:
                chunk.metadata["chunk_index"] = chunk_index
                new_chunks.append(chunk)

        state.final_chunks = new_chunks
