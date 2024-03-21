from langchain.embeddings.openai import OpenAIEmbeddings
from surround import Assembler

from .baseline.character_split_and_overlap_stage import CharacterSplitAndOverlapStage
from .baseline.langchain_pg_vector_indexing_stage import LangchainPGVectorIndexingStage
from .baseline.metadata_injection_stage import MetadataInjectionStage

ASSEMBLIES = [
    Assembler("baseline").set_stages(
        [
            CharacterSplitAndOverlapStage(experiment_name="baseline"),
            MetadataInjectionStage(),
            LangchainPGVectorIndexingStage(embedding_model=OpenAIEmbeddings()),
        ]
    )
]
