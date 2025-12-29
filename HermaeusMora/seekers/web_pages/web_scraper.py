import os
import requests

from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

from utility_scripts.functions import url_to_filename
from utility_scripts.system_logging import setup_logger

# configure logging
logger = setup_logger(__name__)


def fetch_html(url: str):
    """Fetches HTML from a URL and saves it to a local .html file.
     Returns:
        str: The file path to the html,
        otherwise
        int: -1 if the request fails.
    """
    logger.info(f"Fetching {url}")

    response = requests.get(url)

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
    doc = converter.convert(html_file).document

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
    logger.info(f"Chunking > {Path(path_to_doc).name}")

    if not os.path.exists(path_to_doc):
        logger.error(f"{path_to_doc} Does not exist")
        return -1

    # Converting document
    converter = DocumentConverter()
    doc = converter.convert(html_file).document

    # Initialize tokenizer
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

    # Create HybridChunker
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True  # Merge small adjacent chunks
    )

    # Generating chunks
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)

    logger.info(f"Finished Chunking")
    return chunks, tokenizer, chunker


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


def save_chunks(chunks, chunker):
    """Save chunks to file with separators, preserving context and headings."""

    chunk_dir = Path("chunks").resolve()
    chunk_dir.mkdir(exist_ok=True)

    output_path = chunk_dir / f"chunks.txt"

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"{'='*60}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'='*60}\n")
            print(chunk)
            # Use contextualize to preserve headings and metadata
            contextualized_text = chunker.contextualize(chunk=chunk)
            f.write(contextualized_text)
            f.write("\n\n")

    logger.info(f"✓ Chunks saved to: {output_path}")


if __name__ == "__main__":
    html_file = fetch_html("https://en.uesp.net/wiki/Lore:Hermaeus_Mora")
    markdown_file = convert_html(html_file)

    try:
        chunks, tokenizer, chunker = chunk_document(markdown_file)

        # Analyze chunks
        # analyze_chunks(chunks, tokenizer)

        # Save chunks
        save_chunks(chunks, chunker)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
