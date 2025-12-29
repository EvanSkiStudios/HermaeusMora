import os
import requests

from pathlib import Path
from docling.document_converter import DocumentConverter

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

    status_code = f"{url} > Responded with: {response.status_code}"
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

    logger.info(f"Finished > {converted_path}")


if __name__ == "__main__":
    html_file = fetch_html("https://en.uesp.net/wiki/Lore:Hermaeus_Mora")
    convert_html(html_file)
