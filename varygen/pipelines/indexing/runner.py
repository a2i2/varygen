import os

from varygen.config import ExperimentPipelineConfig
from surround import Runner

from .loaders.file_loader import load_dir, load_from_manifest
from .state import IndexingState


class IndexingRunner(Runner):
    """
    Loads the data for indexing.
    """

    def load_data(self, mode, config: ExperimentPipelineConfig):  # pylint: disable=unused-argument
        """
        Load the data for indexing from either the manifest or the content directory directly.
        """

        # Load from either the manifest or the content directory directly
        manifest_path = os.path.join(config.content_dir, "manifest.json")
        loaded_chunks = load_from_manifest(manifest_path, config)
        if not loaded_chunks:
            loaded_chunks = load_dir(config.content_dir, config)

        return IndexingState(loaded_chunks)
