from enum import Enum
from surround import Assembler

from .stages.facts_extraction import FactsExtraction
from .stages.questions_generation import QuestionsGeneration
from .stages.questions_domain_specific_injection import DomainSpecificInjection
from .stages.question_and_chunk_assignment import QuestionChunkAssignment
from .stages.correctness_evals import CorrectnessEvals
from .stages.variations_generator import VariationsGenerator
from .stages.variations_filter import VariationsFilter


class QaGenerationOption(Enum):
    """
    Supporting 3 different pipelines to handle the Question generation
    1. BASELINE
        - Full end to end pipeline from extracting generating quesitons of documents all the way to generate the variations.
    2. QUESTIONS_GENERATION
        - Only run the questions generation from the provided documents.
    3. VARIATIONS_GENERATION
        - Only run the variations generation from the provided question-answer sets.
    """

    BASELINE = "baseline"
    QUESTIONS_GENERATION = "questions_generation"
    VARIATIONS_GENERATION = "variations_generation"


ASSEMBLIES = [
    Assembler(QaGenerationOption.BASELINE.value).set_stages(
        [
            FactsExtraction(),
            QuestionsGeneration(),
            DomainSpecificInjection(),
            QuestionChunkAssignment(),
            CorrectnessEvals(eval_name="questions"),
            VariationsGenerator(),
            VariationsFilter(),
        ]
    ),
    Assembler(QaGenerationOption.QUESTIONS_GENERATION.value).set_stages(
        [
            FactsExtraction(),
            QuestionsGeneration(),
            DomainSpecificInjection(),
            QuestionChunkAssignment(),
            CorrectnessEvals(eval_name="questions"),
        ]
    ),
    Assembler(QaGenerationOption.VARIATIONS_GENERATION.value).set_stages(
        [
            VariationsGenerator(),
            VariationsFilter(),
        ]
    ),
]
