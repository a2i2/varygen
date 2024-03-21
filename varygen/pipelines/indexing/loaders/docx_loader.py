from pathlib import Path

from varygen.common.document import Document
from llama_index.readers.file.docs_reader import DocxReader


def load_docx_file(path: str):
    """
    Load all text from a given Microsoft Word file.
    """

    reader = DocxReader()
    llama_index_docs = reader.load_data(Path(path))
    return [Document.from_llama_index_document(doc) for doc in llama_index_docs]
