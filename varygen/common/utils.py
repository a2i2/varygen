import logging
import operator
import os
from functools import reduce
from typing import Callable, Dict, List, Tuple, TypeVar, Union

import sqlalchemy
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.pgvector import PGVector
from surround import Assembler, BaseConfig

from ..common.document import Document
from ..config import ExperimentPipelineConfig

LOG = logging.getLogger(__name__)


def find(dot_notation_path, obj):
    """
    Find a value in a nested object using dot notation.
    e.g. find("a.b.c", {"a": {"b": {"c": 1}}}) == 1
    """

    return reduce(operator.getitem, dot_notation_path.split("."), obj)


def get_file_path(relative_or_abs_path: str, config: BaseConfig):
    """
    Gets a file path relative to the project root.
    """

    if os.path.isabs(relative_or_abs_path):
        return relative_or_abs_path

    return os.path.join(config.project_root, relative_or_abs_path)


def get_pg_connection_string(config: ExperimentPipelineConfig):
    """
    Get a connection string for the PGVector store using the given configuration.
    """

    return PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=config.postgres_host,
        port=config.postgres_port,
        database=config.postgres_database,
        user=config.postgres_user,
        password=config.postgres_password,
    )


def get_pg_vector_from_config(embedding_model: Embeddings, config: ExperimentPipelineConfig):
    """
    Connect to a PGVector store using the given configuration.
    """

    connection_string = get_pg_connection_string(config)
    return PGVector(connection_string, embedding_model)


def get_all_chunks_from_pg(config: ExperimentPipelineConfig) -> List[Document]:
    """
    Get all chunks from PGVector store using the given configuration.
    """

    # Connect to the PG database.
    connection_string = get_pg_connection_string(config)
    engine = sqlalchemy.create_engine(connection_string)
    connection = engine.connect()

    # Fetch all the chunks from the langchain_pg_embedding table.
    rows = connection.execute(
        sqlalchemy.text("SELECT document as page_content, cmetadata as metadata FROM langchain_pg_embedding")
    ).fetchall()

    # Convert each row to a Document.
    results = []
    for row in rows:
        results.append(Document(page_content=row[0], metadata=row[1]))

    return results


def group_chunks_by_filename(chunks: List[Document]) -> Dict[str, List[Document]]:
    """
    Group chunks by filename.
    """

    results: Dict[str, List[Document]] = {}
    for chunk in chunks:
        location = chunk.metadata["filename"]
        if location not in results:
            results[location] = []

        results[location].append(chunk)

    return results


def shrink_chunk(chunk: str, max_size: int, get_num_tokens: Callable[[str], int]) -> str:
    """
    Shrink a chunk to a given size.
    Uses sentence splitting to ensure the chunk is still valid.
    """

    chunk_size = get_num_tokens(chunk)
    if chunk_size <= max_size:
        return chunk

    LOG.warning(
        "Input chunk is too large (%d tokens), truncating to %d tokens",
        chunk_size,
        max_size,
    )

    chunk_split_by_sentence = chunk.split(".")
    chunk_split_by_sentence = [sentence for sentence in chunk_split_by_sentence if sentence != ""]
    chunk_split_by_sentence = [sentence + "." for sentence in chunk_split_by_sentence]

    chunk = ""
    for sentence in chunk_split_by_sentence:
        if get_num_tokens(chunk + sentence) > max_size:
            break
        chunk += sentence

    return chunk


def combine_chunks(chunks: List[Document], max_size: int, get_num_tokens: Callable[[str], int]) -> str:
    """
    Combine chunks into a single context.
    """

    # Combine chunks into a single context.
    context = ""
    for chunk in chunks:
        # Check if the chunk is too long to add to the context
        if get_num_tokens(context) + get_num_tokens(chunk.page_content) > max_size:
            break

        # Add the chunk to the context
        context += chunk.page_content + "\n\n"

    # At least include one truncated chunk if possible.
    if context == "" and chunks:
        context = shrink_chunk(chunks[0].page_content, max_size, get_num_tokens)

    return context


T = TypeVar("T")


def safe_run_assembler(assembler: Assembler, state: T) -> Union[Tuple[T, None], Tuple[None, Exception]]:
    """
    Run an assembler and catch any exceptions.
    """

    try:
        assembler.run(state)
        return state, None
    except Exception as e:  # pylint: disable=broad-except
        LOG.error("Error running query: %s", e)
        return None, e
