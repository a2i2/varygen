from typing import Optional, List, Tuple
import pandas as pd

from surround import State


class QaGenerationState(State):
    """
    Data passed between stages in the QAs geneation pipeline.
    """

    # Input Variables
    chunks_df: Optional[pd.DataFrame] = None

    # Intermediate values
    # QA Generation
    raw_extraction_output: Optional[List[Tuple[str, str, str, List[str]]]] = None
    facts_df: Optional[pd.DataFrame] = None
    raw_generation_output: Optional[List[List[Tuple[str, str, str, str]]]] = None
    qa_df: Optional[pd.DataFrame] = None
    injection_df: Optional[pd.DataFrame] = None
    original_question_df: Optional[pd.DataFrame] = None
    correctness_evals_input_df: Optional[pd.DataFrame] = None
    correctness_evals_output_df: Optional[pd.DataFrame] = None

    # Variation generation
    variation_df: Optional[pd.DataFrame] = None
    variation_list_output: Optional[List[pd.DataFrame]] = None
    filter_df: Optional[pd.DataFrame] = None
    property_based_eval_df: Optional[pd.DataFrame] = None

    # Final Output Variables
    final_questions_df: Optional[pd.DataFrame] = None
