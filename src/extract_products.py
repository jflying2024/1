"""Extract structured product records from the Zhebao Electrical product catalogue.

The 2021 catalogue (.docx) stores product specs inside multi-column Word tables.
Each product lives in a single cell as a list of bullet points of the form::

    ● Model: KYN28A-12, GZS1-12
    ● Name: AC armored removable metal enclosed switchgear
    ● Rated voltage (Ue): 12kV;
    ● Rated current (Ir): 630A 1250A 1600A;
    ...

This module flattens those cells into one row per product. The well-known
attributes (model, name, voltage, current, standard, dimension ...) are lifted
into their own columns, and every bullet is also preserved verbatim in a single
``raw_parameters`` field so nothing is lost for the downstream LLM step.

Run::

    python src/extract_products.py \
        --docx data/Zhebao_products_catalogue_2021.docx \
        --out-csv data/zhebao_products.csv \
        --out-json data/zhebao_products.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from docx import Document

# Bullet markers used in the source catalogue.
_BULLET_RE = re.compile(r"[●○]\s*")
# A catalogue cell is treated as a product only if it declares a model or name.
_PRODUCT_HINT_RE = re.compile(r"(model|name)\s*:", re.IGNORECASE)

# Map of normalised attribute name -> list of label patterns found in the source.
# The first pattern that matches a bullet's "label:" wins.
_FIELD_PATTERNS: dict[str, list[str]] = {
    "model": [r"model"],
    "name": [r"name"],
    "rated_frequency": [r"rated frequency"],
    "rated_voltage": [r"rated voltage", r"rated input voltage", r"rated voltage ur"],
    "rated_insulation_voltage": [r"rated insulation voltage"],
    "power_frequency_withstand_voltage": [
        r"power freq",
        r"rated power freq",
        r"power frequency withstand",
        r"rated power frequency withstand",
    ],
    "rated_impulse_withstand_voltage": [r"rated impulse withstand"],
    "rated_current": [
        r"rated current",
        r"rated busbar current",
        r"horizontal busbar rated",
        r"circuit current",
    ],
    "short_time_withstand_current": [r"short time withstand"],
    "short_circuit_breaking_current": [
        r"short circuit drop-out",
        r"short circuit breaking",
        r"rated short circuit drop out",
    ],
    "short_circuit_making_current": [
        r"short circuit making",
        r"rated short circuit making",
    ],
    "protection_degree": [r"protect degree", r"protection degree"],
    "standard": [r"standard"],
    "dimension": [r"dimension"],
    "capacity": [r"capacity", r"rated capacity"],
    "scheme_options": [r"scheme options"],
}


@dataclass
class Product:
    """A single catalogue product flattened from a Word table cell."""

    model: str = ""
    name: str = ""
    category: str = ""
    brand: str = "Zhebao"
    voltage_class: str = ""
    rated_frequency: str = ""
    rated_voltage: str = ""
    rated_insulation_voltage: str = ""
    power_frequency_withstand_voltage: str = ""
    rated_impulse_withstand_voltage: str = ""
    rated_current: str = ""
    short_time_withstand_current: str = ""
    short_circuit_breaking_current: str = ""
    short_circuit_making_current: str = ""
    protection_degree: str = ""
    standard: str = ""
    dimension: str = ""
    capacity: str = ""
    scheme_options: str = ""
    raw_parameters: str = ""
    extra_notes: list[str] = field(default_factory=list)


def _split_bullets(cell_text: str) -> list[str]:
    """Split a cell into individual, trimmed bullet strings.

    python-docx joins paragraphs (one per bullet) with newlines. We split on
    newlines and on the bullet marker, but deliberately NOT on "/" because the
    parameter values themselves use slashes (e.g. "AC 50Hz / 60Hz",
    "42/48kV", "2450H *1200/ 1400W").
    """
    parts = re.split(r"[\r\n]+|" + _BULLET_RE.pattern, cell_text)
    return [p.strip(" ;\t") for p in parts if p.strip(" ;\t")]


def _match_field(label: str) -> str | None:
    label = label.strip().lower()
    for field_name, patterns in _FIELD_PATTERNS.items():
        for pat in patterns:
            if re.fullmatch(pat, label) or re.match(pat + r"\b", label):
                return field_name
    return None


def _categorise(name: str, model: str) -> tuple[str, str]:
    """Infer a coarse product category and voltage class for grouping/GEO."""
    text = f"{name} {model}".lower()
    if "transformer" in text and "instrument" not in text and "current" not in text and "voltage transformer" not in text:
        category = "Distribution Transformer"
    elif "circuit breaker" in text or "vcb" in text or "acb" in text or "mccb" in text:
        category = "Circuit Breaker"
    elif "substation" in text:
        category = "Prefabricated Substation"
    elif "load switch" in text or "lbs" in text or "sf6" in text and "switchgear" not in text:
        category = "SF6 Load Break Switch"
    elif "switchgear" in text or "rmu" in text or "switch panel" in text:
        category = "Switchgear / RMU"
    elif "distribution box" in text or "power distribution box" in text or "terminal" in text or "socket" in text:
        category = "Distribution Box"
    elif "transformer" in text or "arrestor" in text or "arrester" in text or "disconnect" in text:
        category = "Outdoor Primary Equipment"
    elif "compensation" in text or "capacitor" in text or "dc power" in text:
        category = "Power Quality / DC Supply"
    else:
        category = "Other"

    # Voltage class from the model suffix or rated voltage text.
    vclass = ""
    m = re.search(r"(\d{1,3}(?:\.\d)?)\s*kv", text)
    if m:
        kv = float(m.group(1))
        if kv >= 40:
            vclass = "High Voltage (40.5kV)"
        elif kv >= 3:
            vclass = "Medium Voltage (3.6-24kV)"
    if not vclass and re.search(r"\b(400v|690v|380v|1000v|low voltage|\blv\b)\b", text):
        vclass = "Low Voltage (<=1kV)"
    return category, vclass


def _detect_brand(model: str, name: str) -> str:
    text = f"{model} {name}".lower()
    for brand in ("siemens", "abb", "schneider"):
        if brand in text:
            return brand.capitalize()
    return "Zhebao"


def parse_cell(cell_text: str) -> Product | None:
    bullets = _split_bullets(cell_text)
    if not bullets:
        return None
    joined = " | ".join(bullets)
    if not _PRODUCT_HINT_RE.search(joined):
        return None

    product = Product(raw_parameters=joined)
    for bullet in bullets:
        if ":" in bullet:
            label, _, value = bullet.partition(":")
            field_name = _match_field(label)
            if field_name and not getattr(product, field_name):
                setattr(product, field_name, value.strip(" ;"))
                continue
        product.extra_notes.append(bullet)

    if not product.model and not product.name:
        return None

    product.brand = _detect_brand(product.model, product.name)
    product.category, product.voltage_class = _categorise(product.name, product.model)
    return product


def extract(docx_path: Path) -> list[Product]:
    document = Document(str(docx_path))
    products: list[Product] = []
    seen: set[str] = set()
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                product = parse_cell(cell.text)
                if product is None:
                    continue
                key = (product.model or product.name).lower()
                if key in seen:
                    continue
                seen.add(key)
                products.append(product)
    return products


def to_records(products: list[Product]) -> list[dict]:
    records = []
    for p in products:
        d = asdict(p)
        d["extra_notes"] = " | ".join(d["extra_notes"])
        records.append(d)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docx",
        type=Path,
        default=Path("data/Zhebao_products_catalogue_2021.docx"),
        help="Path to the source catalogue .docx",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("data/zhebao_products.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("data/zhebao_products.json"))
    args = parser.parse_args()

    products = extract(args.docx)
    records = to_records(products)

    import pandas as pd

    df = pd.DataFrame(records)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    args.out_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Extracted {len(records)} products")
    print(f"  CSV  -> {args.out_csv}")
    print(f"  JSON -> {args.out_json}")
    by_cat: dict[str, int] = {}
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {n}")


if __name__ == "__main__":
    main()
