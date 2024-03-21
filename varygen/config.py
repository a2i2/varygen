from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from omegaconf import MISSING
from surround import BaseConfig, config


@dataclass
class IndexingConfig:
    """
    Indexing pipeline configuration
    """

    # CharacterSplitAndOverlapStage configuration
    max_chunk_characters: Optional[int] = MISSING
    overlap_characters: Optional[int] = MISSING


@dataclass
class VariationGenerationConfig:
    """
    QAs variation generation config
    """

    # OpenAI config
    model_name: Optional[str] = MISSING
    temperature: Optional[float] = MISSING
    frequency_penalty: Optional[float] = MISSING

    # Stage config
    cache_dir_path: Optional[str] = MISSING
    intermediate_output_prefix: Optional[str] = MISSING


@dataclass
class QaGenerationConfig:
    """
    QAs generation config
    """

    # Override facts_override. Default to None.
    # Input an integer number of facts to limit number of fact extraction per document.
    facts_override: Optional[int] = 20
    facts_per_chunk: int = 25  # Number of facts to produce per chunk.

    # OpenAI Config
    # Facts extraction
    extraction_model_name: str = "gpt-4-1106-preview"
    extraction_temperature: float = 0
    extraction_max_token: int = 4096
    extraction_top_p: float = 1.0
    extraction_frequency_penalty: float = 0
    extraction_presence_penalty: float = 1.0
    extraction_timeout: int = 90
    # Duplicate remmoval
    duplicate_removal_model_name: str = "gpt-4-1106-preview"
    duplicate_removal_temperature: float = 0
    duplicate_removal_max_token: int = 4096
    duplicate_removal_top_p: float = 1.0
    duplicate_removal_frequency_penalty: float = 0
    duplicate_removal_presence_penalty: float = 0
    duplicate_removal_timeout: int = 60
    # Questions generation
    generation_removal_model_name: str = "gpt-4-1106-preview"
    generation_removal_temperature: float = 0
    generation_removal_max_token: int = 4096
    generation_removal_top_p: float = 1.0
    generation_removal_frequency_penalty: float = 0
    generation_removal_presence_penalty: float = 0
    generation_removal_timeout: int = 60
    generation_split_size: int = 10
    # Domain injection
    injection_model_name: str = "gpt-4-1106-preview"
    injection_temperature: float = 0
    injection_max_token: int = 4096
    injection_top_p: float = 1.0
    injection_frequency_penalty: float = 0
    injection_presence_penalty: float = 0
    injection_timeout: int = 30
    injection_split_size: int = 10
    # Question chunk assignment
    chunk_assignment_model_name: str = "gpt-4-1106-preview"
    chunk_assignment_temperature: float = 0
    chunk_assignment_max_token: int = 4096
    chunk_assignment_top_p: float = 1.0
    chunk_assignment_frequency_penalty: float = 0
    chunk_assignment_presence_penalty: float = 0
    chunk_assignment_timeout: int = 60
    chunk_assignment_split_size: int = 10
    # Question evaluation
    fetch_k: int = 7


class ExperimentPipeline(str, Enum):
    """
    Experiment pipeline
    """

    INDEXING = "indexing"
    QA_GENERATION = "qa_generation"


@config
@dataclass
class ExperimentPipelineConfig(BaseConfig):
    """
    Experiment platform configuration
    """

    # Experiment name -> Indexing pipeline configuration mapping
    indexing: Dict[str, IndexingConfig] = field(default_factory=dict)

    # Which pipeline to run
    pipeline: ExperimentPipeline = ExperimentPipeline.INDEXING
    assembler: str = "baseline"

    # Use the first runner of the pipeline by default.
    # Doubtful we'll ever need to change this.
    runner: str = "0"

    # Show pipeline information
    status: bool = False

    # Postgres database configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "postgres"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Indexing pipeline configuration
    content_dir: str = "../content-download/content"

    # Evaluation pipeline configuration
    evaluate_csv: str = "input/evaluate.csv"
    estimated_evaluation_output_tokens: int = 20

    # General pipeline configuration (applies to all pipelines)
    parallel: bool = True
    dry_run: bool = False
    gpt4_cost_per_input_token: float = 0.00003
    gpt4_cost_per_output_token: float = 0.00006
    number_of_workers: int = 6  # Adjust this to fit the available CPU cores in your machine

    # QAs generation pipeline config
    qa_generation_csv: str = "input/document_list.csv"
    qa_generation: QaGenerationConfig = field(default_factory=QaGenerationConfig)
    qa_variation_csv: str = "input/original_questions.csv"
    variation_generation: VariationGenerationConfig = field(default_factory=VariationGenerationConfig)
    variation_filter_generation: VariationGenerationConfig = field(default_factory=VariationGenerationConfig)
    output_dir_from_previous_run: Optional[str] = None

    # Output directory will be magically filled in by Hydra
    output_dir: Optional[str] = None
