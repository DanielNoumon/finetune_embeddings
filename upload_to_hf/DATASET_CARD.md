---
language:
- nl
license: cc-by-4.0
task_categories:
- text-retrieval
- sentence-similarity
- feature-extraction
task_ids:
- semantic-similarity-scoring
- document-retrieval
pretty_name: EU AI Act (NL) - Synthetic Query-Chunk Pairs
size_categories:
- 1K<n<10K
tags:
- legal
- regulatory
- dutch
- synthetic
- embedding
- rag
- retrieval
---

# Dataset Card for EU AI Act (NL) - Synthetic Query-Chunk Pairs

## Dataset Description

### Dataset Summary

This dataset contains 2,284 synthetic Dutch query-chunk pairs derived from the Dutch version of the EU Artificial Intelligence Act (Verordening Artificiële Intelligentie). Each pair consists of a realistic user query and the relevant text chunk from the regulation that answers it.

The dataset is designed for **fine-tuning embedding models** for semantic search and retrieval-augmented generation (RAG) applications in the Dutch legal/regulatory domain.

**Key Statistics:**
- **Total pairs:** 2,284
- **Language:** Dutch (nl)
- **Source document:** EU AI Act (Dutch translation)
- **Queries per chunk:** 4 (average)
- **Query length:** 32-344 characters (mean: 127)
- **Chunk length:** 201-4,060 characters (mean: 1,137)

### Supported Tasks

- **Embedding fine-tuning:** Train or fine-tune sentence/document embedding models using Multiple Negatives Ranking Loss (MNRL)
- **Semantic search:** Build search systems for legal/regulatory documents
- **Retrieval-augmented generation (RAG):** Create question-answering systems for the EU AI Act
- **Information retrieval evaluation:** Benchmark embedding models on Dutch legal text

### Languages

Dutch (nl)

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | int | Unique identifier for each query (0-2283) |
| `query` | string | Synthetic Dutch query/question |
| `chunk` | string | Relevant text chunk from the EU AI Act that answers the query |
| `document_name` | string | Source document name ("EU AI Act (NL)") |
| `chunk_id` | int | Identifier of the chunk within the source document |
| `section_type` | string | Type of section: "artikel" (article), "overweging" (recital), or "bijlage" (annex) |
| `hierarchy_path` | string | Location of the chunk in the document structure (e.g., "HOOFDSTUK V > Artikel 52") |

### Data Splits

This dataset does not include predefined splits. Users should create their own train/validation/test splits based on their use case. 

**Recommended approach:** Split at the **chunk level** (not pair level) to prevent data leakage, as multiple queries may reference the same chunk.

### Example

```json
{
  "question_id": 172,
  "query": "Mag markttoezicht broncode van hoog-risico AI opvragen?",
  "chunk": "13. Markttoezichtautoriteiten wordt toegang tot de broncode van het AI-systeem met een hoog risico verleend op een met redenen omkleed verzoek...",
  "document_name": "EU AI Act (NL)",
  "chunk_id": 172,
  "section_type": "artikel",
  "hierarchy_path": "HOOFDSTUK IX > Artikel 74 > Lid 13 + Lid 14"
}
```

## Dataset Creation

### Source Data

#### Initial Data Collection

The source document is the **Dutch translation of the EU Artificial Intelligence Act** (Regulation on Artificial Intelligence), published by the European Union. The regulation establishes harmonized rules for the development, placement on the market, and use of AI systems in the EU.

- **Document:** EU AI Act (Dutch version)
- **Format:** PDF
- **Sections included:** Recitals (overwegingen), Articles (artikelen), and Annexes (bijlagen)

#### Text Preprocessing

The document was processed using a **semantic hierarchical chunking strategy** to preserve legal structure and context:

1. **PDF extraction:** Text extracted from the official PDF using PyMuPDF
2. **Structure parsing:** Document parsed into hierarchical sections:
   - Recitals (numbered considerations)
   - Articles (with paragraphs/leden and sub-paragraphs)
   - Annexes
3. **Semantic chunking:** Text split into chunks respecting:
   - Legal boundaries (articles, paragraphs, recitals)
   - Maximum chunk size: ~1000 tokens (~4000 characters)
   - Minimum chunk size: 50 tokens (with merging of tiny chunks)
4. **Metadata preservation:** Each chunk tagged with:
   - Section type (artikel/overweging/bijlage)
   - Hierarchy path (location in document structure)
   - Unique chunk ID

**Total chunks created:** 573

**Distribution by section type:**
- Articles (artikel): 329 chunks (57.4%)
- Recitals (overweging): 223 chunks (38.9%)
- Annexes (bijlage): 21 chunks (3.7%)

### Question Generation

#### Process

Synthetic queries were generated using **GPT-5-mini** (Azure OpenAI) via the Responses API. For each chunk, 4 diverse Dutch queries were generated to simulate realistic user questions.

**Generation parameters:**
- Model: GPT-5-mini (Azure OpenAI deployment)
- Queries per chunk: 4
- Concurrency: 10 parallel requests
- Total generation time: ~15 minutes (885 seconds)
- Success rate: 99.7% (2,284 / 2,292 expected pairs)

#### Prompt Used

**System Prompt:**
```
Je bent een expert in de EU AI-verordening (EU AI Act) in het Nederlands.

Je taak: gegeven een tekstfragment uit de EU AI Act, genereer {n} diverse zoekquery's in het Nederlands die een gebruiker zou kunnen stellen en waarvoor dit fragment het relevante antwoord bevat.

Regels:
- Schrijf ALLEEN in het Nederlands.
- Elke query moet een andere invalshoek hebben. Wissel VERPLICHT af tussen:
  * Feitelijke vragen ("Welke AI-systemen zijn verboden?")
  * Definitievragen ("Wat wordt bedoeld met hoog-risico AI?")
  * Procedurele vragen ("Hoe voldoe ik aan de transparantieverplichtingen?")
  * Scenariovragen ("Een bedrijf wil gezichtsherkenning inzetten, welke regels gelden?")
- BELANGRIJK: Genereer minstens 1 procedurele vraag ("Hoe...?", "Welke stappen...?") en 1 scenariovraag ("Een bedrijf/organisatie wil...") per {n} queries.
- Varieer de lengte: genereer zowel korte (30-60 tekens) als langere queries (100-150 tekens).
- Query's moeten realistisch zijn — alsof een jurist, beleidsmaker of compliance officer ze zou stellen.
- Verwijs NIET letterlijk naar artikelnummers in de query (de gebruiker kent die nummers vaak niet).
- Antwoord met een JSON-object met een "queries" veld.

Voorbeeld output:
{"queries": ["AI verboden?", "Hoe voldoe ik aan de conformiteitseisen voor hoog-risico AI-systemen?", "Een zorginstelling wil AI gebruiken voor diagnoses, welke verplichtingen gelden?", "Wat wordt bedoeld met transparantieverplichtingen voor AI?"]}
```

**User Prompt Template:**
```
Tekstfragment (bron: EU AI Act NL):
---
{chunk_text}
---

Metadata:
- Type: {section_type}
- Locatie: {hierarchy_path}

Genereer {n} diverse zoekquery's waarvoor bovenstaand fragment het antwoord is.
```

#### Query Diversity

The generated queries exhibit good diversity across multiple dimensions:

**Query types (heuristic classification):**
- Factual ("Welke...?"): ~54%
- Definitional ("Wat...?"): ~27%
- Procedural ("Hoe...?"): ~10%
- Scenario-based (mentions "bedrijf"/"organisatie"): ~40%
- Other: ~9%

**Query length distribution:**
- Short (<60 chars): ~25%
- Medium (60-150 chars): ~35%
- Long (>150 chars): ~40%

**Quality metrics:**
- Unique queries: 2,291 / 2,284 (99.96% unique)
- Queries with explicit article references: 1.9% (low, as intended)

### Quality Control

- Queries are generated to be answerable using **only** the information in the provided chunk
- No external knowledge or context required
- Queries avoid referencing article numbers (users typically don't know these)
- Diversity enforced through explicit prompt instructions

## Considerations for Using the Data

### Intended Use

This dataset is intended for:
- **Fine-tuning embedding models** for Dutch legal/regulatory text retrieval
- **Training semantic search systems** for the EU AI Act
- **Building RAG applications** for legal question-answering
- **Benchmarking** retrieval models on Dutch legal text

### Limitations

1. **Synthetic queries:** All queries are LLM-generated and may not fully represent real user information needs
2. **Single document source:** Limited to the EU AI Act (Dutch version) — may not generalize to other legal domains
3. **Query type imbalance:** Factual and definitional queries are overrepresented compared to procedural queries
4. **No hard negatives:** Dataset contains only positive pairs; hard negatives must be mined separately
5. **Translation quality:** Dependent on the quality of the Dutch translation of the EU AI Act
6. **Temporal scope:** Based on a specific version of the regulation (may become outdated with amendments)

### Bias and Fairness

- Queries generated by GPT-5-mini may reflect biases present in the model's training data
- Query phrasing may favor certain professional contexts (legal, compliance) over layperson language
- No demographic or user diversity in query generation (all synthetic)

### Evaluation Recommendations

When evaluating models trained on this dataset:
- Use **chunk-level splits** (not pair-level) to avoid data leakage
- Evaluate on **information retrieval metrics:** MRR@k, NDCG@k, Recall@k
- Consider testing on **real user queries** if available
- Benchmark against **baseline models** (e.g., multilingual-e5-base, BM25)

## Additional Information

### Licensing

This dataset is released under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

The source document (EU AI Act) is an official EU regulation and is in the public domain under EU law.

### Citation

If you use this dataset in your research or applications, please cite:

```bibtex
@dataset{noumon2026euaiact,
  title={EU AI Act (NL) - Synthetic Query-Chunk Pairs},
  author={Noumon, Daniel},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/[your-username]/eu-ai-act-nl-queries}}
}
```

### Contact

For questions, issues, or feedback about this dataset, please open an issue on the dataset repository or contact the author.

### Acknowledgments

- **Source document:** European Union (EU AI Act - Dutch translation)
- **Query generation:** Azure OpenAI (GPT-5-mini)
- **Chunking and processing:** Custom semantic hierarchical chunking pipeline

### Version History

- **v1.0 (2026-03-09):** Initial release with 2,284 query-chunk pairs
