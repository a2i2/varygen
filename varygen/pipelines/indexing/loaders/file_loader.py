import json
import logging
import os
from glob import glob
from typing import List, Optional

from varygen.common.document import Document
from varygen.common.utils import get_file_path
from varygen.config import ExperimentPipelineConfig

from .html_loader import load_html_file
from .pdf_loader import load_pdf_file
from .pptx_loader import load_pptx_file
from .txt_loader import load_txt_file
from .docx_loader import load_docx_file

LOG = logging.getLogger(__name__)


def load_mp4_file_transcript(path: str) -> List[Document]:
    """
    Loads the transcript of an MP4 file.
    Typically, this is a file with the same name as the MP4 file, but with a .txt extension.
    """

    return load_txt_file(os.path.splitext(path)[0] + ".txt")


loaders = {
    "html": load_html_file,
    "pdf": load_pdf_file,
    "pptx": load_pptx_file,
    "docx": load_docx_file,
    # Enable this once the JS pipeline supports txt files
    # "txt": load_txt_file,
    "mp4": load_mp4_file_transcript,
}


def load_file(path: str, location: Optional[str] = None) -> List[Document]:
    """
    Loads a file from the filesystem.
    Maps the file extension to the appropriate loader.
    """

    extension = path.split(".")[-1]
    documents = loaders[extension](path)

    # Inject metadata required by subsequent steps
    for document in documents:
        document.metadata["filename"] = path.split("/")[-1]
        document.metadata["location"] = location if location else "/".join(path.split("/")[:-1])

    return documents


def load_dir(path: str, config: ExperimentPipelineConfig) -> List[Document]:
    """
    Loads all files in a directory.
    """

    # Get the absolute path, relative to the script root.
    abs_path = get_file_path(path, config)

    LOG.info("Loading files from %s/**/*", path)

    # Find all files in the directory that are eligible for loading
    paths = glob(f"{abs_path}/**/*", recursive=True)
    paths = [path for path in paths if os.path.isfile(path)]
    paths = [path for path in paths if path.split(".")[-1] in loaders]

    # Filter out video transcripts
    mp4_paths = [path for path in paths if path.split(".")[-1] == "mp4"]
    transcription_paths = [os.path.splitext(path)[0] + ".txt" for path in mp4_paths]
    paths = [path for path in paths if path not in transcription_paths]

    LOG.info("Found %d eligible files", len(paths))
    return [doc for path in paths for doc in load_file(path)]


def load_from_manifest(manifest_path: str, config: ExperimentPipelineConfig) -> List[Document]:
    """
    Loads files from a manifest file generated by the /scripts/content-download script.
    """

    abs_path = get_file_path(manifest_path, config)
    if not os.path.exists(abs_path):
        LOG.info("Manifest not found at %s", manifest_path)
        return []

    LOG.info("Loading files from %s", manifest_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    content_dir = os.path.dirname(abs_path)

    def check_existence(path):
        ext = path.split(".")[-1]
        # Check for video transcripts instead of the video itself
        # There is a bug where the video filename differs from the transcript filename slightly.
        if ext == "mp4":
            return os.path.isfile(os.path.splitext(path)[0] + ".txt")
        return os.path.isfile(os.path.join(content_dir, path))

    paths = [file["target_path"] for file in manifest["content"]["downloaded"]]
    paths = [(path, os.path.join(content_dir, path)) for path in paths]
    paths = [(location, path) for location, path in paths if check_existence(path)]
    paths = [(location, path) for location, path in paths if path.split(".")[-1] in loaders]

    LOG.info("Found %d eligible files", len(paths))

    return [doc for location, path in paths for doc in load_file(path, location)]
