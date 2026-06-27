"""Post-generation validation for the GEO pipeline.

Heavy electrical equipment lives and dies by parameter accuracy, and LLMs
occasionally hallucinate units (writing 10V instead of 10kV) or silently drop a
rating. This module performs a program-level cross-check between the SOURCE
parameters (from the catalogue) and the GENERATED Markdown so a human only has
to review the products that actually fail.

The check is intentionally conservative: it extracts the salient numeric tokens
(values with electrical units) from the source fields and flags any that do not
appear verbatim in the generated text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numeric token followed by an electrical unit, e.g. 12kV, 1250A, 31.5kA, 50Hz,
# 2200mm, 3150kVA, IP40. Case-insensitive on the unit.
_UNIT_TOKEN_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(kV|kA|kVA|kvar|Ah|mm|Hz|A|V|W)\b",
    re.IGNORECASE,
)
_IP_TOKEN_RE = re.compile(r"\bIP\s?(\d{1,2}[Xx]?)\b")

# Fields whose numeric ratings are safety-critical and must survive intact.
_CRITICAL_FIELDS = (
    "rated_voltage",
    "rated_insulation_voltage",
    "power_frequency_withstand_voltage",
    "rated_impulse_withstand_voltage",
    "rated_current",
    "short_time_withstand_current",
    "short_circuit_breaking_current",
    "short_circuit_making_current",
    "capacity",
    "protection_degree",
)


@dataclass
class ValidationResult:
    model: str
    missing_values: list[str] = field(default_factory=list)
    suspicious_units: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_values and not self.suspicious_units

    def summary(self) -> str:
        if self.ok:
            return f"[OK]   {self.model}: all critical ratings present"
        bits = []
        if self.missing_values:
            bits.append(f"missing {self.missing_values}")
        if self.suspicious_units:
            bits.append(f"suspicious {self.suspicious_units}")
        return f"[WARN] {self.model}: " + "; ".join(bits)


def _normalise(token: tuple[str, str]) -> str:
    value, unit = token
    return f"{value}{unit.lower()}"


def _extract_unit_tokens(text: str) -> set[str]:
    tokens = {_normalise(m) for m in _UNIT_TOKEN_RE.findall(text)}
    tokens |= {f"ip{m.lower()}" for m in _IP_TOKEN_RE.findall(text)}
    return tokens


def validate_product(product: dict, generated_markdown: str) -> ValidationResult:
    """Cross-check a generated page against the source parameters."""
    result = ValidationResult(model=str(product.get("model") or product.get("name", "?")))
    generated_tokens = _extract_unit_tokens(generated_markdown)
    generated_lower = generated_markdown.lower()

    for fld in _CRITICAL_FIELDS:
        source_value = str(product.get(fld, "") or "")
        if not source_value:
            continue
        for token in _extract_unit_tokens(source_value):
            if token not in generated_tokens:
                result.missing_values.append(f"{fld}:{token}")

    # Unit-confusion heuristic: a value the source gave in kV that appears in the
    # output as bare V (the classic 10kV -> 10V hallucination). We only inspect
    # source tokens whose unit is genuinely kV, so legitimately LV products rated
    # in 400V/690V are not flagged.
    for value, unit in _UNIT_TOKEN_RE.findall(str(product.get("rated_voltage", ""))):
        if unit.lower() != "kv":
            continue
        if re.search(rf"(?<![\w.]){re.escape(value)}\s*V\b", generated_markdown):
            if f"{value}kv" not in generated_lower:
                result.suspicious_units.append(f"rated_voltage {value}kV rendered as {value}V")

    return result


def validate_batch(pairs: list[tuple[dict, str]]) -> list[ValidationResult]:
    return [validate_product(p, md) for p, md in pairs]
