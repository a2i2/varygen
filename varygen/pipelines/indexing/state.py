import json
from typing import List, Optional

from varygen.common.document import Document
from surround import State


class IndexingState(State):
    """
    Data passed between stages in the indexing pipeline.
    """

    # Input Variables
    loaded_chunks: List[Document]

    # Final Output Variables
    final_chunks: Optional[List[Document]] = None

    def __init__(self, loaded_chunks):
        super().__init__()
        self.loaded_chunks = loaded_chunks

    def to_json(self):
        """
        Convert the state to JSON.
        """

        loaded_chunks = self.loaded_chunks if self.loaded_chunks else []
        final_chunks = self.final_chunks if self.final_chunks else []

        return json.dumps(
            {
                "loaded_chunks": [chunk.to_json() for chunk in loaded_chunks],
                "final_chunks": [chunk.to_json() for chunk in final_chunks],
            }
        )
