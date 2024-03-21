from varygen.common.document import Document


def load_txt_file(path: str):
    """
    Load all text from a given TXT file.
    """

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(page_content=text, metadata={})]
