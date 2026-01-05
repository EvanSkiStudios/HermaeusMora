import os
import requests
import json

from pathlib import Path
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

from apocrypha.EpistolaryAcumen import RetainKnowledge
from seekers.web_pages.test_reformat import clean_markdown_file
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


def fetch_html(url: str):
    """Fetches HTML from a URL and saves it to a local .html file.
     Returns:
        str: The file path to the html,
        otherwise
        int: -1 if the request fails.
    """
    logger.info(f"Fetching {url}")

    response = requests.get(url, headers=HEADERS)

    status_code = f"{url} || Responded with: {response.status_code}"
    if response.status_code != 200:
        logger.error(status_code)
        return -1

    logger.info(status_code)

    save_dir = Path("downloads").resolve()
    save_dir.mkdir(exist_ok=True)

    path = save_dir / f"{url_to_filename(url)}.html"

    with path.open("wb") as f:
        f.write(response.content)

    return str(path)


def convert_html(path_to_html):
    """Converts HTML to a formated MarkDown File
     Returns:
        str: The file path to the markdown file,
        otherwise
        int: -1 if the html file does not exist.
    """
    logger.info(f"Converting > {Path(path_to_html).name}")

    if not os.path.exists(path_to_html):
        logger.error(f"{path_to_html} Does not exist")
        return -1

    converter = DocumentConverter()
    doc = converter.convert(path_to_html).document

    markdown_dir = Path("markdowns").resolve()
    markdown_dir.mkdir(exist_ok=True)

    path = markdown_dir / Path(path_to_html).name
    converted_path = path.with_suffix(".md")

    markdown_text = doc.export_to_markdown()

    with converted_path.open("w", encoding="utf-8") as f:
        f.write(markdown_text)

    logger.info(f"Finished > {Path(converted_path).name}")
    return str(converted_path)


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
    html_file = fetch_html("https://en.uesp.net/wiki/Lore:Hermaeus_Mora")
    markdown_file = convert_html(html_file)

    cleaned_markdown = clean_markdown_file(markdown_file)

    try:
        file_name, chunks, tokenizer, chunker = chunk_document(cleaned_markdown)

        # Analyze chunks
        # analyze_chunks(chunks, tokenizer)

        # Save chunks
        json_chunks = save_chunks(file_name, chunks, chunker)
        RetainKnowledge(json_chunks)

        # clean up
        os.remove(html_file)
        # os.remove(markdown_file)
        os.remove(json_chunks)

    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
