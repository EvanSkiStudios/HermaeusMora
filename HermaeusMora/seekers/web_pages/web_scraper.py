import re
import requests

from utility_scripts.system_logging import setup_logger

# configure logging
logger = setup_logger(__name__)


def url_to_filename(url: str) -> str:
    # Remove scheme
    name = re.sub(r'^https?://', '', url)
    # Replace invalid filename characters with underscores
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    # Replace remaining non-alphanumeric characters with underscores
    name = re.sub(r'[^A-Za-z0-9._-]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def fetch_html(url):
    url = "https://en.uesp.net/wiki/Lore:Hermaeus_Mora"
    response = requests.get(url)

    status_code = f"{url}\nResponded with: {response.status_code}"
    if response.status_code != 200:
        logger.error(status_code)
        return -1

    logger.info(status_code)

    file_name = url_to_filename(url)
    with open(f"{file_name}.html", "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    fetch_html("")
