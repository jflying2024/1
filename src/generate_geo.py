"""Batch GEO content generator for the Zhebao Electrical product catalogue.

Pipeline
--------
1. Ingest the structured product table (CSV or JSON produced by
   ``extract_products.py``) with pandas.
2. For every product, build a GEO system+user prompt (see ``prompts.py``).
3. Call an OpenAI-compatible chat completion API with rate limiting and
   automatic retry/back-off (production grade).
4. Cross-validate the generated copy against the source ratings
   (see ``validate.py``) to catch unit hallucinations.
5. Write one Markdown file per product (optionally HTML) plus a run report.

The script also ships an OFFLINE engine (``--engine offline``) that renders a
deterministic GEO template directly from the parameters with no API key. This is
used for tests, demos and previewing the page structure before spending tokens.

Examples
--------
    # Offline demo (no API key required), 3 products:
    python src/generate_geo.py --engine offline --limit 3

    # Full run against OpenAI:
    export OPENAI_API_KEY=sk-...
    python src/generate_geo.py --engine openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

# Allow ``python src/generate_geo.py`` as well as ``python -m src.generate_geo``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import (  # noqa: E402
    build_user_prompt,
    domain_for_category,
    get_system_prompt,
)
from validate import validate_product  # noqa: E402

# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

# Columns that hold real parameter values (fed to the model as JSON).
_PARAM_COLUMNS = [
    "model",
    "name",
    "category",
    "brand",
    "voltage_class",
    "rated_frequency",
    "rated_voltage",
    "rated_insulation_voltage",
    "power_frequency_withstand_voltage",
    "rated_impulse_withstand_voltage",
    "rated_current",
    "short_time_withstand_current",
    "short_circuit_breaking_current",
    "short_circuit_making_current",
    "protection_degree",
    "standard",
    "dimension",
    "capacity",
    "scheme_options",
    "extra_notes",
]


def load_products(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    elif path.suffix.lower() in (".csv",):
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    return df.fillna("")


def product_to_json(row: pd.Series) -> str:
    data = {
        col: str(row[col]).strip()
        for col in _PARAM_COLUMNS
        if col in row and str(row[col]).strip()
    }
    return json.dumps(data, ensure_ascii=False)


def slugify(value: str, fallback: str) -> str:
    value = (value or "").strip() or fallback
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return safe.strip("_") or fallback


# ---------------------------------------------------------------------------
# Generation engines
# ---------------------------------------------------------------------------


class OfflineEngine:
    """Deterministic, no-API GEO renderer used for demos, tests and previews.

    It produces the exact section structure the LLM is asked to produce, copying
    the numeric ratings verbatim so the output always passes validation. This
    doubles as the canonical specification of the target page layout.
    """

    name = "offline"

    _LABELS = [
        ("rated_frequency", "Rated frequency"),
        ("rated_voltage", "Rated voltage (Ue/Ur)"),
        ("rated_insulation_voltage", "Rated insulation voltage (Ui)"),
        ("power_frequency_withstand_voltage", "Power-frequency withstand voltage (1 min)"),
        ("rated_impulse_withstand_voltage", "Rated impulse withstand voltage"),
        ("rated_current", "Rated current (Ir)"),
        ("short_time_withstand_current", "Short-time withstand current (Icw)"),
        ("short_circuit_breaking_current", "Short-circuit breaking current"),
        ("short_circuit_making_current", "Short-circuit making current"),
        ("capacity", "Rated capacity"),
        ("protection_degree", "Protection degree (IP)"),
        ("standard", "Standard"),
        ("dimension", "Dimensions (HxWxD)"),
    ]

    def generate(self, product: dict, system_prompt: str, user_prompt: str) -> str:
        from prompts import (
            DOMAIN_SCENARIOS,
            DOMAIN_STANDARDS,
            domain_for_category as _dfc,
        )

        model = product.get("model", "").strip() or "(model n/a)"
        name = product.get("name", "").strip() or "Electrical product"
        domain = _dfc(product.get("category", ""))

        rows = [
            f"| {label} | {str(product[key]).strip()} |"
            for key, label in self._LABELS
            if product.get(key)
        ]
        spec_table = "\n".join(rows) if rows else "| Parameter | Value |\n| --- | --- |"

        standards = product.get("standard", "").strip()
        std_lines = [f"- {s}" for s in DOMAIN_STANDARDS[domain]]
        if standards:
            std_lines.insert(0, f"- Declared in catalogue: **{standards}**")

        scenarios = DOMAIN_SCENARIOS[domain]
        scen_lines = [f"- {s.capitalize()}." for s in scenarios]

        dim = product.get("dimension", "").strip()
        logistics = (
            f"Catalogue panel dimensions are **{dim}**. "
            if dim
            else "Crate to the panel's outer dimensions. "
        )
        logistics += (
            "Estimate (verify before booking): floor-standing MV/LV panels are "
            "typically exported in plywood sea-worthy crates; a standard 40ft HC "
            "container loads roughly 6-10 single panels depending on width. "
            "Bundle spare parts and operating tools in a separate labelled crate."
        )

        return f"""# {model} — {name}

> {name} ({model}) from Zhebao Electrical (Hangzhou) Group, a Siemens/ABB/Schneider \
licensed manufacturer since 1993. Rated for {product.get('rated_voltage', 'its stated voltage')} \
service and built to {standards or 'the applicable IEC standard'}.

## Technical Specifications

| Parameter | Value |
| --- | --- |
{spec_table}

## International Standards & Compliance

{chr(10).join(std_lines)}

## Application Scenarios

{chr(10).join(scen_lines)}

## GEO-Targeted FAQ

**Q: What is the {model} and what is it used for?**
A: It is a {name.lower()} rated at {product.get('rated_voltage', 'the stated voltage')}, used in {scenarios[0]}.

**Q: Which international standard does the {model} comply with?**
A: {standards or DOMAIN_STANDARDS[domain][0]}.

**Q: What rated current options are available?**
A: {product.get('rated_current', 'Contact Zhebao for the available current ratings.')}

**Q: What is the protection (IP) degree?**
A: {product.get('protection_degree', 'Refer to the project specification.')}

**Q: Can it be supplied for high-altitude / plateau projects?**
A: Yes — derated variants are available; share the site altitude and ambient conditions when inquiring.

## Export & Logistics (estimate)

{logistics}
"""


class OpenAIEngine:
    """OpenAI-compatible chat-completions engine with retry/back-off."""

    name = "openai"

    def __init__(self, model: str, base_url: str | None, temperature: float,
                 max_retries: int = 5):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it, or use --engine offline."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def generate(self, product: dict, system_prompt: str, user_prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as err:  # noqa: BLE001 - retry on any transient API error
                last_err = err
                backoff = min(2 ** attempt + random.uniform(0, 1), 30)
                print(
                    f"    API error (attempt {attempt}/{self.max_retries}): {err}. "
                    f"Retrying in {backoff:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
        raise RuntimeError(f"Generation failed after {self.max_retries} retries: {last_err}")


def build_engine(args: argparse.Namespace):
    if args.engine == "offline":
        return OfflineEngine()
    return OpenAIEngine(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )


# ---------------------------------------------------------------------------
# Optional HTML rendering
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}}
th{{background:#f3f4f6}}
blockquote{{border-left:4px solid #2563eb;margin:1rem 0;padding:.4rem 1rem;background:#f8fafc;color:#334155}}
h1{{border-bottom:2px solid #2563eb;padding-bottom:.3rem}}
</style></head><body>
{body}
</body></html>
"""


def markdown_to_html(md_text: str, title: str) -> str:
    try:
        import markdown  # type: ignore

        body = markdown.markdown(md_text, extensions=["tables"])
    except Exception:  # markdown package optional
        from html import escape

        body = "<pre>" + escape(md_text) + "</pre>"
    return _HTML_TEMPLATE.format(title=title, body=body)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    df = load_products(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} products from {args.input} (engine={args.engine})")

    engine = build_engine(args)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        product = {col: row[col] for col in df.columns}
        model = str(product.get("model", "")).strip()
        name = str(product.get("name", "")).strip()
        label = model or name or f"product_{i}"
        print(f"[{i}/{len(df)}] {label}")

        domain = domain_for_category(str(product.get("category", "")))
        system_prompt = get_system_prompt(domain)
        user_prompt = build_user_prompt(product_to_json(row))

        try:
            markdown_text = engine.generate(product, system_prompt, user_prompt)
        except Exception as err:  # noqa: BLE001
            print(f"    FAILED: {err}", file=sys.stderr)
            report.append({"model": label, "status": "error", "detail": str(err)})
            continue

        slug = slugify(model or name, f"product_{i}")
        md_path = out_dir / f"{slug}.md"
        md_path.write_text(markdown_text, encoding="utf-8")

        if args.html:
            html_path = out_dir / f"{slug}.html"
            html_path.write_text(
                markdown_to_html(markdown_text, f"{model} — {name}"), encoding="utf-8"
            )

        validation = validate_product(product, markdown_text)
        print("    " + validation.summary())
        report.append(
            {
                "model": label,
                "status": "ok" if validation.ok else "warn",
                "file": str(md_path),
                "missing_values": validation.missing_values,
                "suspicious_units": validation.suspicious_units,
            }
        )

        if args.engine != "offline" and args.sleep > 0 and i < len(df):
            time.sleep(args.sleep)

    report_path = out_dir / "generation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for r in report if r["status"] == "ok")
    warn = sum(1 for r in report if r["status"] == "warn")
    err = sum(1 for r in report if r["status"] == "error")
    print(f"\nDone. ok={ok} warn={warn} error={err}. Report -> {report_path}")
    return 1 if err else 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/zhebao_products.csv"),
                   help="CSV/JSON/XLSX produced by extract_products.py")
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--engine", choices=["offline", "openai"], default="offline")
    p.add_argument("--model", default="gpt-4o-mini", help="Model id for the openai engine")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""),
                   help="Override base URL (e.g. DeepSeek/Azure compatible endpoint)")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--sleep", type=float, default=1.5,
                   help="Seconds to sleep between API calls (rate limiting)")
    p.add_argument("--limit", type=int, default=0, help="Process only first N products")
    p.add_argument("--html", action="store_true", help="Also emit an HTML file per product")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
