from pathlib import Path

from varygen.common.document import Document
from llama_index.readers.file.docs_reader import PDFReader


def load_pdf_file(path: str):
    """
    Load all text from a given PDF file.
    """

    # Parse all pages from the PDF using LlamaIndex.
    reader = PDFReader()
    llama_index_docs = reader.load_data(Path(path))
    docs = [Document.from_llama_index_document(doc) for doc in llama_index_docs]

    # Combine all pages into a single document
    page_content = ""
    for doc in docs:
        page_content += doc.page_content

    # Generate metadata for the document
    metadata = docs[0].metadata
    del metadata["page_label"]

    return [Document(page_content=page_content, metadata=metadata)]
