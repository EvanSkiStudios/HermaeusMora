import re


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