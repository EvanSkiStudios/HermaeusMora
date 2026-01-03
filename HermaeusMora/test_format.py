import re


def to_markdown(text: str) -> str:
    if not text or not text.strip():
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n")]

    # Drop empty leading lines
    while lines and not lines[0]:
        lines.pop(0)

    if not lines:
        return ""

    # Heading from first line
    heading = lines[0].split(":", 1)[-1].strip()
    lines[0] = f"# {heading}"

    body_lines = lines[1:]

    # Remove known non-content markers
    body_lines = [
        l for l in body_lines
        if l.lower() not in {"contents"}
    ]

    # Remove citations
    body = "\n".join(body_lines)
    body = re.sub(r"\[\[\d+\]\]\([^)]+\)", "", body)

    # Remove markdown links but keep text
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)

    # Remove escaped wiki paths
    body = re.sub(r"\\wiki\\[^\s)]+", "", body)

    # Normalize whitespace
    body = re.sub(r"[ \t]+", " ", body)

    # Reflow text:
    paragraphs = []
    buffer = []

    for line in body.split("\n"):
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue

        buffer.append(line)

    if buffer:
        paragraphs.append(" ".join(buffer))

    body = "\n\n".join(p.strip() for p in paragraphs)

    # Fix spacing before punctuation
    body = re.sub(r"\s+([,.;:])", r"\1", body)

    return f"{lines[0]}\n\n{body}".strip()


if __name__ == "__main__":
    raw_text = """Lore:Hermaeus Mora
Contents
Always lurking, he is the void and the ever-seeing eyes.
[[10]](#cite_note-PRIMA-10)
:585 Mora has been called the wisest of the Daedric Princes, with a mind as old as
[Tamriel](\\wiki\\Lore:Tamriel)
and a body of slime,
[[7]](#cite_note-DFHermaeus-7)
though he describes himself as
*"the riddle unsolveable. The door unopenable. The book unreadable. The question unanswerable."*
[[35]](#cite_note-ESOFlooded-36)
Unlike most Princes, Hermaeus Mora typically does not take on a humanoid form, manifesting instead as black clouds of varied, grotesque assemblages of eyes, tentacles, and claws, or a featureless purple vortex known as the Wretched Abyss,
[[36]](#cite_note-DFAppearance-37)
[[37]](#cite_note-SRAppearance-38)"""

    md = to_markdown(raw_text)
    print(md)

