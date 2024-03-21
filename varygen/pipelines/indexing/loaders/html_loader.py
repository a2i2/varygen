from pathlib import Path

from varygen.common.document import Document
from llama_index.readers.file.html_reader import HTMLTagReader


def load_html_file(path: str):
    """
    Read all text from the <body> tag of the given HTML file.
    """

    reader = HTMLTagReader("body")
    llama_index_docs = reader.load_data(Path(path))
    return [Document.from_llama_index_document(doc) for doc in llama_index_docs]
