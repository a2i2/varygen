import os
from typing import Callable, List

import hydra
from surround import Assembler, Runner, Surround

from .config import ExperimentPipeline, ExperimentPipelineConfig
from .pipelines.indexing.assembler_index import ASSEMBLIES as INDEXING_ASSEMBLIES
from .pipelines.indexing.runner import IndexingRunner
from .pipelines.qa_generation.assembler_index import (
    ASSEMBLIES as QA_GENERATION_ASSEMBLIES,
)
from .pipelines.qa_generation.runner import QaGenerationRunner


@hydra.main(config_name="config")
def main(config: ExperimentPipelineConfig) -> None:
    """
    Main entry point for the Experiment platform.
    """

    pipeline_to_runner_func = {
        ExperimentPipeline.INDEXING: run_indexing,
        ExperimentPipeline.QA_GENERATION: run_qa_generation,
    }

    runner = pipeline_to_runner_func.get(config.pipeline)

    if not runner:
        raise ValueError(f"Unknown pipeline '{config.pipeline}'")

    runner(config)


def run_indexing(config: ExperimentPipelineConfig):
    """
    Runs the indexing pipeline.
    """

    response = run_pipeline(
        IndexingRunner,
        INDEXING_ASSEMBLIES,
        config,
        "indexing",
        "Indexing pipeline for the Experiment platform",
    )

    if response:
        # Save the state
        state, _, _ = response
        with open("indexing_state.json", "w", encoding="utf-8") as file:
            file.write(state.to_json())


def run_qa_generation(config: ExperimentPipelineConfig):
    """
    Runs the QA generation pipeline.
    """

    run_pipeline(
        QaGenerationRunner,
        QA_GENERATION_ASSEMBLIES,
        config,
        "qa_generation",
        "Generate question-anser sets for the Experiment platform",
    )


def run_pipeline(
    runner_cls: Callable[[], Runner],
    assemblies: List[Assembler],
    config: ExperimentPipelineConfig,
    pipeline_name: str,
    description: str,
):
    """
    Runs a pipeline.
    """

    pipeline = Surround(
        [runner_cls()],
        assemblies,
        config,
        f"varygen/{pipeline_name}",
        description,
        os.path.dirname(os.path.dirname(__file__)),
    )

    if config.status:
        pipeline.show_info()
        return None

    return pipeline.run(config.runner, config.assembler, "predict")


if __name__ == "__main__":
    main(None)
