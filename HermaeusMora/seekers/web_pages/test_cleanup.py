from bs4 import BeautifulSoup
import re
import os


def clean_wikipedia_html_file(file_path: str) -> str:
    """
    Cleans a Wikipedia-style HTML file for LLM input:
    - Removes scripts, styles, navs, references, tables, footers.
    - Removes "See also", "References", and "External links" sections entirely.
    - Keeps headings and paragraph text as section markers.
    - Normalizes whitespace.

    Args:
        file_path (str): Path to the local HTML file.

    Returns:
        str: Cleaned, structured text suitable for LLM input.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "sup", "table", "nav", "footer"]):
        tag.decompose()

    # Remove sections like "See also", "References", "External links"
    for heading in soup.find_all(re.compile("^h[1-6]$")):
        heading_text = heading.get_text(strip=True).lower()
        if heading_text in ["see also", "references", "external links"]:
            # Remove the heading and all following siblings until the next heading of same or higher level
            next_node = heading
            level = int(heading.name[1])
            while next_node:
                nxt = next_node.find_next_sibling()
                if nxt and re.match("^h[1-6]$", nxt.name) and int(nxt.name[1]) <= level:
                    break
                next_node.decompose()
                next_node = nxt

    # Extract remaining headings and paragraphs
    content = []
    for element in soup.find_all(["h1", "h2", "h3", "p"]):
        text = element.get_text(separator=" ", strip=True)
        if text:
            content.append(text)

    # Join paragraphs with double newlines for readability
    cleaned_text = "\n\n".join(content)

    # Normalize whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


# Example usage
if __name__ == "__main__":
    html_file_path = "C:\\Users\\reale\\Documents\\GameMakeing\\Development\\Projects\\discord_bot\\HermaeusMora\\HermaeusMora\\seekers\\web_pages\\downloads\\en.uesp.net_wiki_Lore_Hermaeus_Mora.html"
    cleaned_text = clean_wikipedia_html_file(html_file_path)
    print(cleaned_text)