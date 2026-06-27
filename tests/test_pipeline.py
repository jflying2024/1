"""Smoke tests for the GEO pipeline (no network / API key required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_geo import OfflineEngine, product_to_json, slugify  # noqa: E402
from prompts import ProductDomain, domain_for_category, get_system_prompt  # noqa: E402
from validate import validate_product  # noqa: E402

SAMPLE = {
    "model": "KYN28A-12",
    "name": "AC armored removable metal enclosed switchgear",
    "category": "Switchgear / RMU",
    "brand": "Zhebao",
    "rated_voltage": "12kV",
    "power_frequency_withstand_voltage": "42/48kV",
    "rated_current": "630A 1250A 1600A",
    "short_time_withstand_current": "25kA 31.5kA 40kA",
    "protection_degree": "IP40, door open IP2x",
    "standard": "IEC62271-200",
    "dimension": "2260H *650W *1450D (mm)",
}


def test_domain_mapping():
    assert domain_for_category("Switchgear / RMU") is ProductDomain.SWITCHGEAR
    assert domain_for_category("Distribution Transformer") is ProductDomain.TRANSFORMER
    assert domain_for_category("nonsense") is ProductDomain.GENERIC


def test_system_prompt_mentions_standards():
    prompt = get_system_prompt(ProductDomain.SWITCHGEAR)
    assert "IEC 62271" in prompt
    assert "NO marketing fluff" in prompt


def test_offline_engine_preserves_all_ratings():
    md = OfflineEngine().generate(SAMPLE, "", "")
    assert md.startswith("# KYN28A-12")
    for token in ("12kV", "1250A", "31.5kA", "IEC62271-200"):
        assert token in md
    result = validate_product(SAMPLE, md)
    assert result.ok, result.summary()


def test_validation_detects_dropped_value():
    bad_md = "# X\n\nNo numbers here at all.\n"
    result = validate_product(SAMPLE, bad_md)
    assert not result.ok
    assert any("rated_voltage" in m for m in result.missing_values)


def test_validation_flags_kv_to_v_hallucination():
    bad_md = "# X\n\n| Rated voltage | 12V |\n"
    result = validate_product(SAMPLE, bad_md)
    assert result.suspicious_units


def test_lv_voltage_not_flagged():
    lv = {"model": "GCS", "name": "AC LV switchgear", "category": "Distribution Box",
          "rated_voltage": "400V/ 690V"}
    md = OfflineEngine().generate(lv, "", "")
    result = validate_product(lv, md)
    assert not result.suspicious_units


def test_product_to_json_and_slug():
    import pandas as pd

    row = pd.Series(SAMPLE)
    js = product_to_json(row)
    assert "KYN28A-12" in js
    assert slugify("KYN28A-12, GZS1-12", "x") == "KYN28A-12__GZS1-12"
