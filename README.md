# Zhebao Electrical — GEO Content Automation

Batch-generate **high-information-density, GEO-optimised English product pages**
for the Zhebao Electrical (Hangzhou) Group catalogue.

The core idea: take **structured technical parameters (input)** and, through a
prompt constrained by GEO rules, turn them into **scenario-rich, AI-citable
marketing copy (output)** — at catalogue scale.

```
Catalogue .docx  ──►  extract_products.py  ──►  data/zhebao_products.{csv,json}
                                                      │
                                                      ▼
                          prompts.py (GEO system prompt, per domain)
                                                      │
                                                      ▼
        generate_geo.py  ──►  LLM (OpenAI-compatible) or offline engine
                                                      │
                          validate.py (rating cross-check)  ──►  output/*.md (+ .html)
```

## What's in here

| Path | Purpose |
| --- | --- |
| `src/extract_products.py` | Parse the catalogue `.docx` tables → one structured row per product. |
| `src/prompts.py` | GEO system prompt, per-domain IEC standards & application scenarios. |
| `src/generate_geo.py` | Batch generator: pandas ingestion, rate-limit + retry, validation, MD/HTML output. |
| `src/validate.py` | Cross-checks generated copy against source ratings (catches `10kV → 10V` hallucinations). |
| `tools/geo_preview.html` | Interactive browser tool to tune the page structure / prompt per product before a full run. |
| `data/` | Source catalogue + extracted `zhebao_products.csv` / `.json` (**76 products**). |
| `output/samples/` | Curated example generated pages. |
| `tests/` | Offline smoke tests (no API key needed). |

## Quick start

```bash
pip install -r requirements.txt

# 1. (Re)extract the structured dataset from the catalogue
python src/extract_products.py \
    --docx data/Zhebao_products_catalogue_2021.docx \
    --out-csv data/zhebao_products.csv \
    --out-json data/zhebao_products.json

# 2a. Preview the page structure with NO API key (deterministic offline engine)
python src/generate_geo.py --engine offline --limit 5 --html

# 2b. Full GEO run against an LLM
export OPENAI_API_KEY=sk-...
python src/generate_geo.py --engine openai --model gpt-4o-mini --sleep 1.5 --html
```

Generated Markdown (and optional HTML) land in `output/`, one file per product,
plus `output/generation_report.json` summarising validation results.

### Using a different provider

`generate_geo.py` speaks the OpenAI chat-completions API, so any compatible
endpoint works (DeepSeek, Azure OpenAI, local gateways):

```bash
export OPENAI_API_KEY=...
python src/generate_geo.py --engine openai \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat
```

## The interactive preview / config tool

`tools/geo_preview.html` lets you debug the GEO output **before** spending tokens
on a few hundred products. Serve the repo and open it:

```bash
python -m http.server 8000
# open http://localhost:8000/tools/geo_preview.html
```

You can select any product, toggle which GEO sections to emit (spec table,
standards, scenarios, FAQ, logistics), flip the GEO constraints (ban fluff / lock
numbers), pick application-scenario tags, and see the rendered page, the raw
Markdown, the exact system prompt, and a live validation badge — all client-side.

## GEO rules enforced

Every page is built to be cited by AI answer engines (ChatGPT, Perplexity,
Gemini), not to read like a brochure:

1. **No marketing fluff** — facts and concrete values only.
2. **Numeric ratings copied verbatim** — `validate.py` flags any dropped or
   unit-mangled value (`10kV` must never become `10V`).
3. **Markdown spec table** of all core parameters.
4. **International standards** cited per domain (IEC 62271 for switchgear,
   IEC 60076 for transformers, IEC 61439 for LV assemblies, …).
5. **Application scenarios** tied to concrete emerging-market pain points
   (mining, unstable grids, emergency replacement, …).
6. **GEO-targeted FAQ** matching long-tail buyer queries.
7. **Export & logistics estimate** (sea-worthy crating / container loading) to
   reverse-fill the trade data factories usually omit.

## Tests

```bash
python -m pytest -q
```
