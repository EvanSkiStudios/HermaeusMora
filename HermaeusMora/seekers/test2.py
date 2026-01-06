import re
import os

SECTION_BLACKLIST = {
    "see also",
    "references",
    "external links",
    "notes",
    "further reading",
    "navigation menu"
}


def clean_markdown_file(file_path: str,
                        remove_code_blocks: bool = True,
                        remove_tables: bool = True) -> str:
    """
    Cleans a Markdown file for LLM input.

    - Removes images, URLs, footnotes, HTML blocks
    - Optionally removes code blocks and tables
    - Preserves headings as section markers
    - Removes blacklisted sections entirely
    - Normalizes whitespace

    Returns LLM-ready plain text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove HTML blocks
    text = re.sub(r"<[^>]+>", "", text)

    # Remove images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Convert links to plain text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove footnote references
    text = re.sub(r"\[\^.+?\]", "", text)
    text = re.sub(r"^\[\^.+?\]:.*$", "", text, flags=re.MULTILINE)

    # Remove code blocks
    if remove_code_blocks:
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove tables
    if remove_tables:
        text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)

    lines = text.splitlines()
    cleaned = []
    skip_section = False

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            heading_text = heading_match.group(2).strip().lower()

            if heading_text in SECTION_BLACKLIST:
                skip_section = True
                continue
            else:
                skip_section = False
                cleaned.append(f"\nSection: {heading_match.group(2).strip()}")
                continue

        if skip_section:
            continue

        if line.strip():
            cleaned.append(line.strip())

    # Normalize whitespace
    output = "\n".join(cleaned)
    output = re.sub(r"\n{3,}", "\n\n", output)
    output = re.sub(r"[ \t]+", " ", output)

    return output.strip()


# Example usage
if __name__ == "__main__":
    path = "C:\\Users\\reale\\Documents\\GameMakeing\\Development\\Projects\\discord_bot\\HermaeusMora\\HermaeusMora\\seekers\\web_pages\\markdowns\\en.uesp.net_wiki_Lore_Hermaeus_Mora.md"
    cleaned_text = clean_markdown_file(path)
    print(cleaned_text)