import os
import logging
from typing import List, Tuple
import pandas as pd

from varygen.common import utils
from surround import Runner

from .state import QaGenerationState
from .assembler_index import QaGenerationOption
from ...common.document import Document  # pylint: disable=E0402

LOG = logging.getLogger(__name__)

ORIGINAL_INPUT_HEADERS = ["question", "answer", "chunk_numbers", "chunks", "filename"]


class QaGenerationRunner(Runner):
    """
    Loads the data for indexing.
    """

    def load_data(self, _, __):
        """
        Empty stub - we'll do this in the `run` method.
        """

    def run(self, _: str):
        """
        Runs the QA generation pipeline.
        """
        if not self.assembler:
            raise ValueError("Cannot run the QA Generation pipeline without an assembler.")

        selected_assembly = QaGenerationOption(self.assembler.assembler_name)

        state = QaGenerationState()
        if selected_assembly == QaGenerationOption.VARIATIONS_GENERATION:
            # Process variation stages directly
            variation_input_csv = self.assembler.config.qa_variation_csv
            if not variation_input_csv:
                raise ValueError("Cannot run Variation Generation pipeline without the input CSV.")

            variation_input_csv = utils.get_file_path(variation_input_csv, self.assembler.config)
            if not os.path.exists(variation_input_csv):
                LOG.error("Variation Generation Input CSV does not exist: %s", variation_input_csv)
                raise ValueError("Variation Generation Input CSV does not exist")

            # Load input data (question - answer pairs)
            original_df = pd.read_csv(variation_input_csv)
            # Only collect required columns
            original_df = original_df[ORIGINAL_INPUT_HEADERS]
            state.original_question_df = original_df
        else:
            # Process question generation or full pipeline
            question_generation_input_csv = self.assembler.config.qa_generation_csv
            if not question_generation_input_csv:
                raise ValueError("Cannot run QA generation pipeline without the input CSV.")

            question_generation_input_csv = utils.get_file_path(question_generation_input_csv, self.assembler.config)
            if not os.path.exists(question_generation_input_csv):
                LOG.error("QA Generation Input CSV does not exist: %s", question_generation_input_csv)
                raise ValueError("QA Generation Input CSV does not exist")

            # Load list of documents to extract questions
            input_df = pd.read_csv(question_generation_input_csv)

            docs = utils.group_chunks_by_filename(utils.get_all_chunks_from_pg(self.assembler.config))
            print(docs)
            filtered_docs: List[Tuple[str, List[Document]]] = [
                (key, doc) for key, doc in docs.items() if self._filename_in_list(key, input_df["filename"].tolist())
            ]
            LOG.info("Found %s document(s) to extract questions from.", len(filtered_docs))

            # Parsing the input to be a DF
            chunks_list: List[Tuple[str, str, str, str]] = []
            for key, doc in filtered_docs:
                for chunk in doc:
                    chunks_list.append(
                        (
                            chunk.page_content,
                            key,
                            chunk.metadata["chunk_index"],
                            self._get_keywords_from_location(key, input_df),
                        )
                    )
            chunks_df = pd.DataFrame(chunks_list, columns=["chunk", "filename", "index", "keywords_to_insert"])
            state.chunks_df = chunks_df

        # Initialise the assembler.
        self.assembler.init_assembler()
        self.assembler.run(state)

    def _filename_in_list(self, filename: str, document_list: List[str]) -> bool:
        for doc in document_list:
            if filename.endswith(doc):
                return True
        return False

    def _get_keywords_from_location(self, location: str, input_df: pd.DataFrame) -> str:
        filename = os.path.basename(location)
        keywords = input_df[input_df["filename"] == filename]["keywords_to_insert"].tolist()
        if len(keywords) == 0:
            raise RuntimeError(f"No keywords can be found for '{location}'")
        return keywords[0]
