import os
import requests
import json
import re

import trafilatura
from bs4 import BeautifulSoup

from pathlib import Path
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

from apocrypha.EpistolaryAcumen import RetainKnowledge
from utility_scripts.functions import url_to_filename
from utility_scripts.system_logging import setup_logger

# configure logging
logger = setup_logger(__name__)

# Load Env
load_dotenv()

HEADERS = {
    "User-Agent": os.getenv("USER_AGENT"),
    "Accept": "text/html",
    "Accept-Language": "en-US,en;q=0.5"
}


CITATION_RE = re.compile(r"\[\s*\d+\s*\]")


def remove_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def extract_main_content(html: str, source_url: str | None = None) -> str:
    """
    Attempt main-content extraction using trafilatura.
    Falls back to DOM-based extraction if needed.
    """

    # --- Primary: Trafilatura ---
    extracted = trafilatura.extract(
        html,
        url=source_url,
        include_comments=False,
        include_tables=False,
        include_links=False,
        favor_precision=True
    )

    if extracted and len(extracted) > 1000:
        return extracted.strip()

    # --- Fallback: DOM-based (MediaWiki / UESP / Wikipedia) ---
    soup = BeautifulSoup(html, "lxml")

    # common MediaWiki container
    content = soup.find("div", id="mw-content-text")

    if content:
        text = content.get_text("\n", True)  # type: ignore[arg-type]
        if len(text) > 500:
            return text

    # --- Final fallback: body text ---
    body = soup.body.get_text("\n", strip=True) if soup.body else ""
    return body


def fetch_and_extract(url: str) -> str | int:
    """Fetch URL and return extracted main content text."""
    logger.info(f"Fetching and extracting {url}")

    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        logger.error(f"{url} || Responded with: {response.status_code}")
        return -1

    content = extract_main_content(response.text, source_url=url)

    if not content or len(content) < 500:
        logger.error("Extraction failed or content too small")
        return -1

    return content


def save_markdown(text: str, name: str) -> str:
    markdown_dir = Path("markdowns").resolve()
    markdown_dir.mkdir(exist_ok=True)

    path = markdown_dir / f"{name}.md"

    with path.open("w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"Saved markdown > {path.name}")
    return str(path)


def heuristic_cleanup(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 40:
            continue
        if sum(c.isalnum() for c in line) / max(len(line), 1) < 0.6:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)



def chunk_document(path_to_doc):
    """Chunks document into relevant parts
     Returns:
        file_name, chunks, tokenizer used, chunker used
    """

    file_name = Path(path_to_doc).stem
    logger.info(f"Chunking > {file_name}")

    if not os.path.exists(path_to_doc):
        logger.error(f"{path_to_doc} Does not exist")
        return -1

    # Converting document
    converter = DocumentConverter()
    doc = converter.convert(path_to_doc).document

    # Initialize tokenizer
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

    # Create HybridChunker
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,  # Merge small adjacent chunks
        always_emit_headings=False
    )

    # Generating chunks
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)

    logger.info(f"Finished Chunking")
    return file_name, chunks, tokenizer, chunker


def analyze_chunks(chunks, tokenizer):
    """Analyze and display chunk statistics."""

    total_tokens = 0
    chunk_sizes = []

    for i, chunk in enumerate(chunks):
        # Get text content
        text = chunk.text
        tokens = tokenizer.encode(text)
        token_count = len(tokens)

        total_tokens += token_count
        chunk_sizes.append(token_count)

    # Summary statistics
    logger.info("SUMMARY STATISTICS")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Average tokens per chunk: {total_tokens / len(chunks):.1f}")
    logger.info(f"Min tokens: {min(chunk_sizes)}")
    logger.info(f"Max tokens: {max(chunk_sizes)}")

    # Token distribution
    logger.info(f"Token distribution:")
    ranges = [(0, 128), (128, 256), (256, 384), (384, 512)]
    for start, end in ranges:
        count = sum(1 for size in chunk_sizes if start <= size < end)
        logger.info(f"  {start}-{end} tokens: {count} chunks")


def save_chunks(file_name, chunks, chunker):
    """Save chunks to json, preserving context and headings."""

    chunk_dir = Path("chunks").resolve()
    chunk_dir.mkdir(exist_ok=True)

    output_path = chunk_dir / f"{file_name}__chunks.json"

    with output_path.open("w", encoding="utf-8") as f:
        f.write("[\n")  # start JSON array
        for i, chunk in enumerate(chunks):
            contextualized_text = chunker.contextualize(chunk=chunk)

            chunk_data = {
                "chunk_id": i,
                "content": contextualized_text
            }

            json.dump(chunk_data, f, ensure_ascii=False, indent=4)
            if i < len(chunks) - 1:
                f.write(",\n")  # comma between items
            else:
                f.write("\n")  # last item
        f.write("]")  # end JSON array

    logger.info(f"✓ Chunks saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    url = "https://en.uesp.net/wiki/Lore:Hermaeus_Mora"

    extracted_text = fetch_and_extract(url)
    if extracted_text == -1:
        raise RuntimeError("Failed to extract content")

    extracted_text = remove_citations(extracted_text)
    extracted_text = heuristic_cleanup(extracted_text)

    file_name = url_to_filename(url)
    markdown_file = save_markdown(extracted_text, file_name)

    try:
        file_name, chunks, tokenizer, chunker = chunk_document(markdown_file)
        json_chunks = save_chunks(file_name, chunks, chunker)
        RetainKnowledge(json_chunks)

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
