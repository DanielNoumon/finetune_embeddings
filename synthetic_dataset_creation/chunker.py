"""
Semantic hierarchical chunker for Dutch EU regulation PDFs.

Supports any regulation that follows the standard EU structure:
  Recitals (Overwegingen) → Chapters (Hoofdstukken) → Articles (Artikelen) → Annexes (Bijlagen)

Produces two JSONL outputs per document:
  - chunks_with_context.jsonl   (contextual header prepended to text)
  - chunks_without_context.jsonl (raw chunk text only)

Each chunk carries rich metadata for downstream RAG and synthetic dataset generation.

Usage:
  Set DOC in the config section at the bottom of this file:
    DOC = "all"         # Chunk all configured documents
    DOC = "eu_ai_act"   # Chunk only the EU AI Act
    DOC = "gdpr"        # Chunk only the GDPR
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_CHUNK_TOKENS = 50
MAX_CHUNK_TOKENS = 1000
OVERLAP_TOKENS = 50
APPROX_CHARS_PER_TOKEN = 4  # rough estimate for Dutch text

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DocumentConfig:
    """Configuration for chunking a specific EU regulation PDF."""
    name: str                        # short key, e.g. "eu_ai_act", "gdpr"
    source_label: str                # used in chunk metadata, e.g. "eu_ai_act_NL"
    display_name: str                # used in context headers, e.g. "EU AI Act (NL)"
    pdf_path: Path
    output_dir: Path
    footer_patterns: list[re.Pattern] = field(default_factory=list)
    definitions_article: int = 3     # which article contains definitions
    has_annexes: bool = True


# Pre-configured documents
DOCUMENTS: dict[str, DocumentConfig] = {
    "eu_ai_act": DocumentConfig(
        name="eu_ai_act",
        source_label="eu_ai_act_NL",
        display_name="EU AI Act (NL)",
        pdf_path=PROJECT_ROOT / "data" / "documents" / "eu_ai_act_NL.pdf",
        output_dir=PROJECT_ROOT / "data" / "chunks" / "eu_ai_act",
        footer_patterns=[
            re.compile(r"PB L van \d{1,2}\.\d{1,2}\.\d{4}"),
            re.compile(r"^NL$", re.MULTILINE),
            re.compile(r"ELI:\s*http\S+"),
            re.compile(r"\d{1,3}/144"),
        ],
        definitions_article=3,
        has_annexes=True,
    ),
    "gdpr": DocumentConfig(
        name="gdpr",
        source_label="gdpr_NL",
        display_name="AVG (NL)",
        pdf_path=PROJECT_ROOT / "data" / "documents" / "gdpr_NL.pdf",
        output_dir=PROJECT_ROOT / "data" / "chunks" / "gdpr",
        footer_patterns=[
            re.compile(r"Publicatieblad van de Europese Unie"),
            re.compile(r"^NL$", re.MULTILINE),
            re.compile(r"L\s+119/\d+"),
            re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}"),
        ],
        definitions_article=4,
        has_annexes=False,
    ),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int = 0
    text: str = ""
    source: str = "eu_ai_act_NL"
    section_type: str = ""          # recital | article | annex
    chapter: str = ""
    chapter_number: str = ""
    section: str = ""               # Afdeling, if any
    article_number: Optional[int] = None
    article_title: str = ""
    paragraph_number: Optional[int] = None
    annex_number: str = ""
    annex_title: str = ""
    hierarchy_path: str = ""
    token_estimate: int = 0
    context_header: str = ""        # contextual prefix


# ---------------------------------------------------------------------------
# Text extraction & cleaning
# ---------------------------------------------------------------------------

def extract_text(pdf_path: Path, footer_patterns: list[re.Pattern]) -> str:
    """Extract full text from the PDF, cleaning headers/footers per page."""
    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        text = page.get_text()
        text = _clean_page(text, footer_patterns)
        pages.append(text)
    doc.close()
    return "\n".join(pages)


def _clean_page(text: str, footer_patterns: list[re.Pattern]) -> str:
    """Remove recurring headers, footers, and fix column-break artefacts."""
    for pat in footer_patterns:
        text = pat.sub("", text)
    # Fix hyphenated line breaks from two-column layout (word-\n continuation)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Structural parsing
# ---------------------------------------------------------------------------

# Regex patterns for structural elements
# Recital pattern: (N) alone on a line, or (N) at line start followed by text.
# GDPR recitals 100+ are inline: "(100) Teneinde..." instead of "(100)\n"
RE_RECITAL = re.compile(
    r"^\((\d+)\)\s*$|^\((\d+)\)\s+(?=[A-Z])", re.MULTILINE
)
RE_CHAPTER = re.compile(
    r"(HOOFDSTUK\s+[IVXLC]+)\s*\n"
    r"\s*((?![Aa][Ff][Dd][Ee][Ll][Ii][Nn][Gg])"
    r"[A-Z][A-Z\- ,À-Þ]{2,}"
    r"(?:\n(?![Aa][Ff][Dd][Ee][Ll][Ii][Nn][Gg])"
    r"[A-Z][A-Z\- ,À-Þ]{2,})*)",
    re.MULTILINE,
)
RE_SECTION = re.compile(r"(Afdeling\s+\d+)\s*\n\s*(.+?)(?=\n)", re.MULTILINE)
RE_ARTICLE = re.compile(r"^(Artikel\s+(\d+))\s*$", re.MULTILINE)
RE_PARAGRAPH = re.compile(r"^(\d+)\.\s*$", re.MULTILINE)
RE_ANNEX = re.compile(r"^(BIJLAGE\s+([IVXLC]+))\s*(?:\n(.+?))?(?=\n)", re.MULTILINE)


def _find_recitals_end(text: str) -> int:
    """Return the char index where recitals end (first Artikel appears)."""
    m = re.search(r"^Artikel\s+1\s*$", text, re.MULTILINE)
    if m:
        # Walk back to find HOOFDSTUK I which precedes Artikel 1
        hfd = text.rfind("HOOFDSTUK I", 0, m.start())
        return hfd if hfd != -1 else m.start()
    return 0


def _find_annexes_start(text: str) -> int:
    """Return the char index where annexes begin."""
    m = re.search(r"^BIJLAGE\s+I\s*$", text, re.MULTILINE)
    return m.start() if m else len(text)


# ---------------------------------------------------------------------------
# Recital parsing
# ---------------------------------------------------------------------------

def _parse_recitals(text: str, config: DocumentConfig) -> list[Chunk]:
    """Parse numbered recitals from the preamble."""
    chunks: list[Chunk] = []
    raw_matches = list(RE_RECITAL.finditer(text))

    # Deduplicate: footnote refs like (1), (2) may appear before
    # the actual recitals. Keep only the last match per number,
    # then filter to a contiguous ascending sequence.
    last_by_num: dict[int, re.Match] = {}
    for m in raw_matches:
        num = int(m.group(1) or m.group(2))
        last_by_num[num] = m
    matches = sorted(last_by_num.values(), key=lambda m: m.start())

    for i, m in enumerate(matches):
        start = m.end()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches) else len(text)
        )
        body = text[start:end].strip()
        # group(1) for standalone, group(2) for inline
        recital_num = int(m.group(1) or m.group(2))
        hierarchy = f"Overwegingen > ({recital_num})"
        chunk = Chunk(
            text=body,
            source=config.source_label,
            section_type="overweging",
            paragraph_number=recital_num,
            hierarchy_path=hierarchy,
            context_header=f"{config.display_name} > Overwegingen > Overweging ({recital_num}):",
        )
        if _estimate_tokens(body) > MAX_CHUNK_TOKENS:
            chunks.extend(_split_long_text(chunk, config))
        else:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Article parsing
# ---------------------------------------------------------------------------

def _parse_articles(text: str, config: DocumentConfig) -> list[Chunk]:
    """Parse articles, splitting long ones at paragraph (lid) boundaries."""
    chunks: list[Chunk] = []

    # Track current chapter / section context
    current_chapter = ""
    current_chapter_num = ""
    current_section = ""

    # Build a combined event stream: chapters, sections, articles
    events: list[tuple[int, str, dict]] = []

    for m in RE_CHAPTER.finditer(text):
        chapter_num = m.group(1).strip()
        chapter_title = re.sub(r"\s+", " ", m.group(2).strip())
        events.append((m.start(), "chapter", {"num": chapter_num, "title": chapter_title}))

    for m in RE_SECTION.finditer(text):
        events.append((m.start(), "section", {"label": m.group(1).strip(), "title": m.group(2).strip()}))

    article_matches = list(RE_ARTICLE.finditer(text))
    for m in article_matches:
        events.append((m.start(), "article", {"match": m, "num": int(m.group(2))}))

    events.sort(key=lambda e: e[0])

    # Process each article with its context
    for idx, (pos, etype, data) in enumerate(events):
        if etype == "chapter":
            current_chapter = f"{data['num']} — {data['title']}"
            current_chapter_num = data["num"]
            current_section = ""
        elif etype == "section":
            current_section = f"{data['label']} — {data['title']}"
        elif etype == "article":
            art_num = data["num"]
            art_match = data["match"]

            # Determine article body end
            art_body_start = art_match.end()
            art_body_end = len(text)
            for next_pos, next_type, _ in events[idx + 1:]:
                if next_type in ("article", "chapter"):
                    art_body_end = next_pos
                    break

            art_body = text[art_body_start:art_body_end].strip()

            # Extract article title (first bold line after "Artikel N")
            title_match = re.match(r"^([^\n]+)", art_body)
            art_title = title_match.group(1).strip() if title_match else ""

            # Build hierarchy prefix
            hierarchy_parts = [current_chapter]
            if current_section:
                hierarchy_parts.append(current_section)
            hierarchy_parts.append(f"Artikel {art_num}")
            if art_title:
                hierarchy_parts[-1] += f" — {art_title}"

            # Try splitting by paragraph (lid) numbers
            para_chunks = _split_article_by_paragraphs(
                art_body, art_num, art_title,
                current_chapter, current_chapter_num,
                current_section, hierarchy_parts, config,
            )
            chunks.extend(para_chunks)

    return chunks


def _split_article_by_paragraphs(
    body: str,
    art_num: int,
    art_title: str,
    chapter: str,
    chapter_num: str,
    section: str,
    hierarchy_parts: list[str],
    config: DocumentConfig,
) -> list[Chunk]:
    """Split an article body into paragraph-level chunks."""
    para_matches = list(RE_PARAGRAPH.finditer(body))

    # Special case: definitions article — split by numbered items
    if art_num == config.definitions_article:
        return _split_definitions(body, art_title, chapter, chapter_num, section, hierarchy_parts, config)

    # If no paragraph markers or article is short enough, keep as single chunk
    if len(para_matches) <= 1 or _estimate_tokens(body) <= MAX_CHUNK_TOKENS:
        hierarchy = " > ".join(hierarchy_parts)
        header = f"{config.display_name} > {hierarchy}:"
        chunk = Chunk(
            text=body,
            source=config.source_label,
            section_type="artikel",
            chapter=chapter,
            chapter_number=chapter_num,
            section=section,
            article_number=art_num,
            article_title=art_title,
            hierarchy_path=hierarchy,
            context_header=header,
        )
        if _estimate_tokens(body) > MAX_CHUNK_TOKENS:
            return _split_long_text(chunk, config)
        return [chunk]

    chunks: list[Chunk] = []
    # Include any text before the first numbered paragraph (article title / intro)
    intro_text = body[:para_matches[0].start()].strip()

    for i, pm in enumerate(para_matches):
        para_num = int(pm.group(1))
        start = pm.start()
        end = para_matches[i + 1].start() if i + 1 < len(para_matches) else len(body)
        para_text = body[start:end].strip()

        # Prepend intro text to the first paragraph
        if i == 0 and intro_text:
            para_text = intro_text + "\n\n" + para_text

        hierarchy = " > ".join(hierarchy_parts) + f" > Lid {para_num}"
        header = f"{config.display_name} > {hierarchy}:"

        chunk = Chunk(
            text=para_text,
            source=config.source_label,
            section_type="artikel",
            chapter=chapter,
            chapter_number=chapter_num,
            section=section,
            article_number=art_num,
            article_title=art_title,
            paragraph_number=para_num,
            hierarchy_path=hierarchy,
            context_header=header,
        )

        # If still too long, do sentence-level splitting
        if _estimate_tokens(para_text) > MAX_CHUNK_TOKENS:
            chunks.extend(_split_long_text(chunk, config))
        else:
            chunks.append(chunk)

    return chunks


def _split_definitions(
    body: str,
    art_title: str,
    chapter: str,
    chapter_num: str,
    section: str,
    hierarchy_parts: list[str],
    config: DocumentConfig,
) -> list[Chunk]:
    """Split a definitions article by numbered definition items."""
    chunks: list[Chunk] = []
    def_art = config.definitions_article
    # Definitions are numbered: 1) ... 2) ... up to ~68)
    def_pattern = re.compile(r"^(\d+)\)", re.MULTILINE)
    def_matches = list(def_pattern.finditer(body))

    # Include intro text before first definition
    if def_matches:
        intro = body[:def_matches[0].start()].strip()
        if intro and _estimate_tokens(intro) >= MIN_CHUNK_TOKENS:
            hierarchy = " > ".join(hierarchy_parts) + " > Inleiding"
            chunks.append(Chunk(
                text=intro,
                source=config.source_label,
                section_type="artikel",
                chapter=chapter,
                chapter_number=chapter_num,
                section=section,
                article_number=def_art,
                article_title=art_title,
                hierarchy_path=hierarchy,
                context_header=f"{config.display_name} > {hierarchy}:",
            ))

    for i, dm in enumerate(def_matches):
        def_num = int(dm.group(1))
        start = dm.start()
        end = def_matches[i + 1].start() if i + 1 < len(def_matches) else len(body)
        def_text = body[start:end].strip()

        hierarchy = " > ".join(hierarchy_parts) + f" > Definitie {def_num}"
        header = f"{config.display_name} > {hierarchy}:"

        chunk = Chunk(
            text=def_text,
            source=config.source_label,
            section_type="artikel",
            chapter=chapter,
            chapter_number=chapter_num,
            section=section,
            article_number=def_art,
            article_title=art_title,
            paragraph_number=def_num,
            hierarchy_path=hierarchy,
            context_header=header,
        )

        # Merge tiny definitions with the next one
        if _estimate_tokens(def_text) < MIN_CHUNK_TOKENS and chunks and chunks[-1].article_number == def_art:
            chunks[-1].text += "\n\n" + def_text
            chunks[-1].hierarchy_path += f", {def_num}"
        else:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Annex parsing
# ---------------------------------------------------------------------------

def _parse_annexes(text: str, config: DocumentConfig) -> list[Chunk]:
    """Parse annexes into chunks."""
    chunks: list[Chunk] = []
    annex_matches = list(RE_ANNEX.finditer(text))

    for i, m in enumerate(annex_matches):
        annex_num = m.group(2).strip()
        annex_title = m.group(3).strip() if m.group(3) else ""

        start = m.end()
        end = annex_matches[i + 1].start() if i + 1 < len(annex_matches) else len(text)
        body = text[start:end].strip()

        hierarchy = f"Bijlage {annex_num}"
        if annex_title:
            hierarchy += f" — {annex_title}"
        header = f"{config.display_name} > {hierarchy}:"

        chunk = Chunk(
            text=body,
            source=config.source_label,
            section_type="bijlage",
            annex_number=annex_num,
            annex_title=annex_title,
            hierarchy_path=hierarchy,
            context_header=header,
        )

        # Split large annexes
        if _estimate_tokens(body) > MAX_CHUNK_TOKENS:
            chunks.extend(_split_long_text(chunk, config))
        else:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Chunk splitting / merging utilities
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return len(text) // APPROX_CHARS_PER_TOKEN


def _split_long_text(chunk: Chunk, config: DocumentConfig) -> list[Chunk]:
    """Split an oversized chunk at sentence boundaries with overlap."""
    sentences = re.split(r"(?<=[.;:])\s+|\n{2,}", chunk.text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sub_chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_tokens = _estimate_tokens(sent)
        if current_len + sent_tokens > MAX_CHUNK_TOKENS and current_sentences:
            sub_text = " ".join(current_sentences)
            sub = Chunk(
                text=sub_text,
                source=chunk.source,
                section_type=chunk.section_type,
                chapter=chunk.chapter,
                chapter_number=chunk.chapter_number,
                section=chunk.section,
                article_number=chunk.article_number,
                article_title=chunk.article_title,
                paragraph_number=chunk.paragraph_number,
                annex_number=chunk.annex_number,
                annex_title=chunk.annex_title,
                hierarchy_path=chunk.hierarchy_path,
                context_header=chunk.context_header,
            )
            sub_chunks.append(sub)

            # Overlap: keep last few sentences
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                s_len = _estimate_tokens(s)
                if overlap_len + s_len > OVERLAP_TOKENS:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += s_len
            current_sentences = overlap_sentences
            current_len = overlap_len

        current_sentences.append(sent)
        current_len += sent_tokens

    # Final sub-chunk
    if current_sentences:
        sub_text = " ".join(current_sentences)
        sub = Chunk(
            text=sub_text,
            source=chunk.source,
            section_type=chunk.section_type,
            chapter=chunk.chapter,
            chapter_number=chunk.chapter_number,
            section=chunk.section,
            article_number=chunk.article_number,
            article_title=chunk.article_title,
            paragraph_number=chunk.paragraph_number,
            annex_number=chunk.annex_number,
            annex_title=chunk.annex_title,
            hierarchy_path=chunk.hierarchy_path,
            context_header=chunk.context_header,
        )
        sub_chunks.append(sub)

    # Label sub-chunks
    if len(sub_chunks) > 1:
        for j, sc in enumerate(sub_chunks):
            sc.hierarchy_path += f" (deel {j + 1}/{len(sub_chunks)})"
            sc.context_header = f"{config.display_name} > {sc.hierarchy_path}:"

    return sub_chunks


def _merge_small_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Merge consecutive tiny chunks of the same section type and parent."""
    merged: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        can_merge_back = (
            merged
            and _estimate_tokens(chunk.text) < MIN_CHUNK_TOKENS
            and merged[-1].section_type == chunk.section_type
            and merged[-1].article_number == chunk.article_number
            and merged[-1].annex_number == chunk.annex_number
            and _estimate_tokens(merged[-1].text)
            + _estimate_tokens(chunk.text) <= MAX_CHUNK_TOKENS
        )
        if can_merge_back:
            merged[-1].text += "\n\n" + chunk.text
            merged[-1].hierarchy_path += (
                f" + {chunk.hierarchy_path.split(' > ')[-1]}")
        else:
            merged.append(chunk)

    # Second pass: merge remaining tiny chunks forward
    final: list[Chunk] = []
    skip_next = False
    for i, chunk in enumerate(merged):
        if skip_next:
            skip_next = False
            continue
        if (
            _estimate_tokens(chunk.text) < MIN_CHUNK_TOKENS
            and i + 1 < len(merged)
            and merged[i + 1].section_type == chunk.section_type
            and merged[i + 1].article_number == chunk.article_number
            and _estimate_tokens(chunk.text)
            + _estimate_tokens(merged[i + 1].text)
            <= MAX_CHUNK_TOKENS
        ):
            merged[i + 1].text = chunk.text + "\n\n" + merged[i + 1].text
            merged[i + 1].hierarchy_path = (
                chunk.hierarchy_path.split(' > ')[-1]
                + " + " + merged[i + 1].hierarchy_path)
        else:
            final.append(chunk)
    return final


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def chunk_document(config: DocumentConfig) -> list[Chunk]:
    """Full chunking pipeline. Returns list of Chunk objects."""
    print(f"\n{'='*60}")
    print(f"Chunking: {config.display_name}")
    print(f"  PDF: {config.pdf_path.name}")
    print(f"  Output: {config.output_dir}")
    print(f"{'='*60}")

    print(f"Extracting text from {config.pdf_path.name} ...")
    full_text = extract_text(config.pdf_path, config.footer_patterns)

    recitals_end = _find_recitals_end(full_text)
    annexes_start = _find_annexes_start(full_text) if config.has_annexes else len(full_text)

    recitals_text = full_text[:recitals_end]
    articles_text = full_text[recitals_end:annexes_start]
    annexes_text = full_text[annexes_start:] if config.has_annexes else ""

    print(f"Parsing recitals ({len(recitals_text):,} chars) ...")
    recital_chunks = _parse_recitals(recitals_text, config)

    print(f"Parsing articles ({len(articles_text):,} chars) ...")
    article_chunks = _parse_articles(articles_text, config)

    annex_chunks: list[Chunk] = []
    if config.has_annexes and annexes_text:
        print(f"Parsing annexes ({len(annexes_text):,} chars) ...")
        annex_chunks = _parse_annexes(annexes_text, config)
    else:
        print("No annexes to parse.")

    all_chunks = recital_chunks + article_chunks + annex_chunks

    # Merge tiny chunks
    all_chunks = _merge_small_chunks(all_chunks)

    # Assign IDs and token estimates
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_id = i
        chunk.token_estimate = _estimate_tokens(chunk.text)

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: list[Chunk], output_dir: Path) -> None:
    """Save two JSONL files: with and without contextual headers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    path_with = output_dir / "chunks_with_context.jsonl"
    path_without = output_dir / "chunks_without_context.jsonl"

    with open(path_with, "w", encoding="utf-8") as f_with, \
         open(path_without, "w", encoding="utf-8") as f_without:
        for chunk in chunks:
            record = asdict(chunk)

            # Version WITHOUT context header — raw text
            record_without = {**record, "text": chunk.text}
            del record_without["context_header"]
            f_without.write(json.dumps(record_without, ensure_ascii=False) + "\n")

            # Version WITH context header — prepended
            record_with = {**record, "text": f"{chunk.context_header}\n\n{chunk.text}"}
            del record_with["context_header"]
            f_with.write(json.dumps(record_with, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to:")
    print(f"  WITH context:    {path_with}")
    print(f"  WITHOUT context: {path_without}")


def print_stats(chunks: list[Chunk]) -> None:
    """Print summary statistics about the generated chunks."""
    from collections import Counter
    type_counts = Counter(c.section_type for c in chunks)
    token_vals = [c.token_estimate for c in chunks]

    print("\n=== Chunk Statistics ===")
    print(f"Total chunks: {len(chunks)}")
    for t, count in type_counts.most_common():
        print(f"  {t}: {count}")
    print(f"Token estimates — min: {min(token_vals)}, max: {max(token_vals)}, "
          f"avg: {sum(token_vals) // len(token_vals)}, total: {sum(token_vals):,}")

    # Distribution buckets
    buckets = {"<50": 0, "50-200": 0, "200-500": 0, "500-800": 0, "800-1000": 0, ">1000": 0}
    for t in token_vals:
        if t < 50:
            buckets["<50"] += 1
        elif t < 200:
            buckets["50-200"] += 1
        elif t < 500:
            buckets["200-500"] += 1
        elif t < 800:
            buckets["500-800"] += 1
        elif t <= 1000:
            buckets["800-1000"] += 1
        else:
            buckets[">1000"] += 1
    print("Token distribution:")
    for label, count in buckets.items():
        print(f"  {label}: {count}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CONFIG: Adjust these parameters for your use case
    # -----------------------------------------------------------------------

    # Which document(s) to process: "eu_ai_act", "gdpr", or "all"
    DOC = "all"

    # -----------------------------------------------------------------------

    docs = list(DOCUMENTS.keys()) if DOC == "all" else [DOC]
    for doc_name in docs:
        config = DOCUMENTS[doc_name]
        chunks = chunk_document(config)
        print_stats(chunks)
        save_chunks(chunks, config.output_dir)
