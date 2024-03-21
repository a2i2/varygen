import ast
import logging
import unicodedata
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings

from varygen.common.utils import get_pg_vector_from_config
from varygen.config import ExperimentPipelineConfig
from varygen.pipelines.qa_generation.state import QaGenerationState

from surround import Stage

LOG = logging.getLogger(__name__)

CACHE_FILENAME_SUFFIX = "correctness_evals.csv"


class CorrectnessEvals(Stage):
    # pylint: disable=C0103
    """
    Handles verifying the correct of questions based on the available context in the vector store.
    """

    def __init__(
        self,
        eval_name: str,
        fetch_k: Optional[int] = None,
        question_column: str = "question",
        chunks_column: str = "chunks",
    ):
        self.pg_vector: Optional[PGVector] = None
        self.eval_name: str = eval_name
        self.fetch_k: Optional[int] = fetch_k
        self.question_column: str = question_column
        self.chunks_column: str = chunks_column

    def initialise(self, config: ExperimentPipelineConfig):
        """
        Initialise the stage with the configuration.
        """

        self.pg_vector = get_pg_vector_from_config(OpenAIEmbeddings(), config)
        # Load from config if not provided
        if not self.fetch_k:
            self.fetch_k = config.qa_generation.fetch_k

    def operate(self, state: QaGenerationState, config: ExperimentPipelineConfig):
        """
        Run questions correctness evaluation.
        """
        if state.correctness_evals_input_df is None:
            raise RuntimeError("List of question is required for correctness evaluations.")

        df = state.correctness_evals_input_df.copy()
        LOG.info("Running correctness evaluations for %s questions.", len(df))
        # Prepare evaluation input
        evaluation_input = [pd.DataFrame([row], columns=df.columns) for _, row in df.iterrows()]
        evaluation_output: List[pd.DataFrame] = []
        if config.parallel:
            with ThreadPoolExecutor(max_workers=config.number_of_workers) as executor:
                # Use multiprocessing to process generate the variation
                evaluation_output.extend(executor.map(self._evaluate_relevancy, evaluation_input))
        else:
            evaluation_output.extend([self._evaluate_relevancy(input) for input in evaluation_input])

        if len(evaluation_output) > 0:
            correctness_evals_df = pd.concat(evaluation_output, ignore_index=True)
            output_filename = f"{self.eval_name}_{CACHE_FILENAME_SUFFIX}"
            correctness_evals_df.to_csv(output_filename, index=False)
            LOG.info("Correctness evaluation are saved at '%s'", output_filename)

            # Print summary
            correctness_evals_df["chunk_found"] = correctness_evals_df["chunk_index_found"].apply(
                lambda index_found: index_found > 0
            )
            # Calculate the percentage of rows where chunk is retrieved for the given question.
            percentage_true = correctness_evals_df["chunk_found"].mean() * 100
            LOG.info("Percentage of good questions: %s%%", round(percentage_true, 2))

            state.correctness_evals_output_df = correctness_evals_df
        else:
            LOG.warning("No correctness evaluation to be saved.")

    def _evaluate_relevancy(self, local_df):
        try:
            local_df[self.chunks_column] = local_df[self.chunks_column].apply(ast.literal_eval)
        except ValueError:
            LOG.warning("Skipping literal_eval since values are already correct.")
        local_df["retrived_chunks"] = local_df[self.question_column].apply(self._retrive_chunks)
        local_df["chunk_index_found"] = local_df.apply(self._match_chunks, axis=1)
        return local_df

    def _retrive_chunks(self, question):
        # Retrieve chunks
        doc_and_scores = self.pg_vector.similarity_search_with_score(question, self.fetch_k)
        docs = [doc[0].page_content for doc in doc_and_scores]
        return docs

    def _match_chunks(self, row):
        original_chunks = [unicodedata.normalize("NFKD", chunk) for chunk in row["chunks"]]
        retrived_chunks = row["retrived_chunks"]
        rank = -1
        for index, chunk_content in enumerate(retrived_chunks):
            norm_page_content = unicodedata.normalize("NFKD", chunk_content)
            if norm_page_content in original_chunks:
                rank = index + 1
                break
        return rank
