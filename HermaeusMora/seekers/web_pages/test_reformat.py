import re
from pathlib import Path


def remove_wiki_ui_noise(text: str) -> str:
    text = re.sub(r"\[\s*edit\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[⧼.*?⧽\]", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"^The UESPWiki.*?$", "", text, flags=re.MULTILINE)
    return text


def normalize_headings(text: str) -> str:
    def clean_heading(match):
        hashes = match.group(1)
        title = match.group(2)
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)
        title = re.sub(r"\s+", " ", title).strip()
        return f"{hashes} {title}\n"

    return re.sub(r"^(#{1,6})\s+(.*)$", clean_heading, text, flags=re.MULTILINE)


def fix_infobox(text: str) -> str:
    table_match = re.search(
        r"(\n\|.*?\|\n(?:\|.*?\|\n)+)", text, flags=re.DOTALL
    )
    if not table_match:
        return text

    raw_table = table_match.group(1)
    rows = []

    for line in raw_table.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2:
            key = cells[0]
            value = cells[-1]
            if key and value:
                rows.append((key, value))

    seen = set()
    cleaned_rows = []
    for key, value in rows:
        if key not in seen:
            seen.add(key)
            cleaned_rows.append((key, value))

    new_table = ["| Key | Value |", "|-----|-------|"]
    for key, value in cleaned_rows:
        new_table.append(f"| {key} | {value} |")

    new_table_text = "\n".join(new_table) + "\n"
    return text.replace(raw_table, "\n" + new_table_text + "\n")


def remove_links(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text


def remove_citations(text: str) -> str:
    # Wiki-style and bracketed citations
    text = re.sub(r"\[\[\s*[^\]]+\s*\]\]", "", text)

    # Parenthesized cite_note anchors
    text = re.sub(r"\(\s*#cite_note[^)]*\)", "", text)

    # Bare cite_note fragments
    text = re.sub(r"#cite_note[-\w]+", "", text)

    # Colon page refs like :585
    text = re.sub(r"\s*:\s*\d+", "", text)

    # Remove standalone citation numbers (core fix)
    # Matches numbers that are:
    # - surrounded by whitespace or punctuation
    # - NOT part of a word, date, or ordinal
    text = re.sub(
        r"(?<![\w])\d{1,3}(?![\w])",
        "",
        text
    )

    # Remove citation-only sections
    text = re.sub(
        r"^##+\s+(References|Notes|See Also).*?$.*?(?=^##|\Z)",
        "",
        text,
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )

    return text


def reformat_lists(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[\*\+•]\s+", "- ", line)
        line = re.sub(r"^\s*\d+\.\s+", "- ", line)
        lines.append(line)
    return "\n".join(lines)


def fix_typography(text: str) -> str:
    text = re.sub(r"\*\*\s*(.*?)\s*\*\*", r"**\1**", text)
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"([.,;:])([^\s])", r"\1 \2", text)
    return text


def normalize_spacing(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip() + "\n"


def clean_markdown(text: str) -> str:
    text = remove_wiki_ui_noise(text)
    text = normalize_headings(text)
    text = fix_infobox(text)
    text = remove_links(text)
    text = remove_citations(text)
    text = reformat_lists(text)
    text = fix_typography(text)
    text = normalize_spacing(text)
    return text


def clean_markdown_file(path: str | Path) -> Path:
    path = Path(path)
    cleaned_path = path.with_name(path.stem + "_cleaned.md")

    text = path.read_text(encoding="utf-8")
    cleaned = clean_markdown(text)
    cleaned_path.write_text(cleaned, encoding="utf-8")

    return cleaned_path
