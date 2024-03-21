from pathlib import Path

from varygen.common.document import Document
from llama_index import download_loader

PptxReader = download_loader("PptxReader")


def load_pptx_file(path: str):
    """
    Load all text from a given PPTX file.
    """

    reader = PptxReader()
    llama_index_docs = reader.load_data(Path(path))
    return [Document.from_llama_index_document(doc) for doc in llama_index_docs]
