import logging

from varygen.config import ExperimentPipelineConfig
from surround import Stage

from ..state import IndexingState

LOG = logging.getLogger(__name__)


class MetadataInjectionStage(Stage):
    """
    Injects metadata into the final chunks, such as the name and location of the file the chunk came from.
    """

    def operate(self, state: IndexingState, config: ExperimentPipelineConfig):  # pylint: disable=unused-argument
        """
        Inject metadata into the final chunks.
        """

        if not state.final_chunks:
            LOG.warning("No chunks to inject metadata into")
            return

        for chunk in state.final_chunks:
            original_content = chunk.page_content

            if "filename" in chunk.metadata:
                chunk.page_content = f"Filename: {chunk.metadata['filename']}\n"

            if "location" in chunk.metadata:
                chunk.page_content += f"Location: {' > '.join(chunk.metadata['location'].split('/')[:-1])}\n"

            chunk.page_content += f'"""{original_content}"""'
            chunk.metadata["original_content"] = original_content
