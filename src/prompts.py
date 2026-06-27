"""GEO (Generative Engine Optimization) prompt assets for Zhebao Electrical.

The whole pipeline hinges on this file: structured technical parameters (input)
are turned into high-information-density, scenario-rich English marketing copy
(output) under a set of GEO constraints. Keeping the prompts here -- separate
from the orchestration code -- makes them easy to review, A/B test and version.

Two product families need slightly different standards and pain points, so the
system prompt is parameterised by ``ProductDomain`` and the relevant IEC
standards / application scenarios are injected per product.
"""

from __future__ import annotations

from enum import Enum


class ProductDomain(str, Enum):
    """Coarse product domain used to steer standards and scenarios."""

    SWITCHGEAR = "switchgear"
    TRANSFORMER = "transformer"
    CIRCUIT_BREAKER = "circuit_breaker"
    SUBSTATION = "substation"
    LV_DISTRIBUTION = "lv_distribution"
    GENERIC = "generic"


# Map our extracted ``category`` strings to a GEO domain.
CATEGORY_TO_DOMAIN: dict[str, ProductDomain] = {
    "Switchgear / RMU": ProductDomain.SWITCHGEAR,
    "Circuit Breaker": ProductDomain.CIRCUIT_BREAKER,
    "SF6 Load Break Switch": ProductDomain.SWITCHGEAR,
    "Prefabricated Substation": ProductDomain.SUBSTATION,
    "Distribution Transformer": ProductDomain.TRANSFORMER,
    "Outdoor Primary Equipment": ProductDomain.SWITCHGEAR,
    "Distribution Box": ProductDomain.LV_DISTRIBUTION,
    "Power Quality / DC Supply": ProductDomain.LV_DISTRIBUTION,
}

# Relevant international standards per domain (used to anchor the FAQ / body).
DOMAIN_STANDARDS: dict[ProductDomain, list[str]] = {
    ProductDomain.SWITCHGEAR: [
        "IEC 62271-200 (AC metal-enclosed switchgear, 1-52 kV)",
        "IEC 62271-1 (common specifications for HV switchgear)",
        "IEC 62271-100 (HV alternating-current circuit-breakers)",
    ],
    ProductDomain.CIRCUIT_BREAKER: [
        "IEC 62271-100 (HV AC circuit-breakers)",
        "IEC 62271-1 (common specifications)",
        "IEC 60694 (legacy common clauses)",
    ],
    ProductDomain.TRANSFORMER: [
        "IEC 60076-11 (dry-type power transformers)",
        "IEC 60076-1 (power transformers, general)",
    ],
    ProductDomain.SUBSTATION: [
        "IEC 62271-202 (HV/LV prefabricated substations)",
        "IEC 62271-200 (the enclosed MV switchgear inside)",
    ],
    ProductDomain.LV_DISTRIBUTION: [
        "IEC 61439-1/-2 (low-voltage switchgear and controlgear assemblies)",
        "IEC 60529 (IP ingress protection ratings)",
    ],
    ProductDomain.GENERIC: [
        "IEC 62271 series (HV switchgear)",
        "IEC 61439 series (LV assemblies)",
    ],
}

# Concrete, third-world / emerging-market pain points per domain. These keep the
# 'Application Scenarios' section grounded instead of generic marketing fluff.
DOMAIN_SCENARIOS: dict[ProductDomain, list[str]] = {
    ProductDomain.SWITCHGEAR: [
        "mining and quarry sites with high dust and vibration",
        "unstable grids with frequent voltage sags and reclosing duty",
        "emergency feeder replacement after flood or storm damage",
        "industrial parks needing compact footprint per panel",
    ],
    ProductDomain.CIRCUIT_BREAKER: [
        "frequent switching duty on motor and capacitor banks",
        "retrofit of ageing oil/SF6 breakers in legacy substations",
        "high-altitude derating for plateau projects",
    ],
    ProductDomain.TRANSFORMER: [
        "indoor distribution in hospitals, malls and high-rise where oil is banned",
        "high-humidity coastal and tropical climates",
        "fire-safety-critical sites requiring cast-resin dry type",
    ],
    ProductDomain.SUBSTATION: [
        "rapid rural electrification and temporary construction power",
        "renewable plant collector stations (solar / wind farms)",
        "fast container-style deployment to remote infrastructure",
    ],
    ProductDomain.LV_DISTRIBUTION: [
        "power distribution for factories and commercial buildings",
        "harsh outdoor/site distribution requiring high IP rating",
        "energy-efficiency and power-quality upgrades",
    ],
    ProductDomain.GENERIC: [
        "infrastructure projects in emerging markets",
        "unstable grid conditions and emergency replacement",
    ],
}


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in international electrical-engineering trade and Generative \
Engine Optimization (GEO). You transform raw technical parameters of \
{domain_label} products from Zhebao Electrical (Hangzhou) Group -- a Siemens / \
ABB / Schneider licensed switchgear manufacturer since 1993 -- into \
high-information-density, GEO-compliant English product detail pages that AI \
answer engines (ChatGPT, Perplexity, Gemini) will cite.

NON-NEGOTIABLE OUTPUT REQUIREMENTS:
1. NO marketing fluff. Ban words like "world-class", "best", "leading", \
"high quality". State only technical facts and concrete values.
2. NEVER invent, round or alter a numeric value. Copy ratings (kV, A, kA, mm, \
kVA, Hz, IP) EXACTLY as given. If a value is missing, omit it -- do not guess. \
A "10kV" must never become "10V".
3. Start with an H1 title "<Model> -- <Name>" and a 2-3 sentence factual \
abstract optimised to directly answer "what is <model>" buyer queries.
4. Render ALL core parameters into one clean Markdown specification table with \
columns | Parameter | Value |.
5. Add an "International Standards & Compliance" section citing the relevant \
standards from this list where applicable: {standards}.
6. Add an "Application Scenarios" section that links specific product features \
to concrete buyer pain points such as: {scenarios}. Be specific, not generic.
7. Add a "GEO-Targeted FAQ" with 4-6 Q&A pairs phrased as the long-tail \
questions international buyers actually ask AI engines (procurement, \
compatibility, standards, lead time, spare parts, altitude/climate derating).
8. Add an "Export & Logistics" note: based on the product dimensions and \
weight class, estimate sea-worthy export crating and 20ft/40ft container \
loading guidance. Clearly label these as estimates.
9. Output clean GitHub-flavoured Markdown ONLY. No HTML, no code fences around \
the whole document, no preamble or sign-off.
"""


def get_system_prompt(domain: ProductDomain) -> str:
    """Build the GEO system prompt for a given product domain."""
    labels = {
        ProductDomain.SWITCHGEAR: "medium/high-voltage switchgear & RMU",
        ProductDomain.CIRCUIT_BREAKER: "vacuum circuit breaker",
        ProductDomain.TRANSFORMER: "distribution transformer",
        ProductDomain.SUBSTATION: "prefabricated substation",
        ProductDomain.LV_DISTRIBUTION: "low-voltage distribution",
        ProductDomain.GENERIC: "electrical power equipment",
    }
    return SYSTEM_PROMPT_TEMPLATE.format(
        domain_label=labels[domain],
        standards="; ".join(DOMAIN_STANDARDS[domain]),
        scenarios="; ".join(DOMAIN_SCENARIOS[domain]),
    )


def domain_for_category(category: str) -> ProductDomain:
    return CATEGORY_TO_DOMAIN.get(category, ProductDomain.GENERIC)


def build_user_prompt(product_json: str) -> str:
    """Wrap a product's JSON parameters into the user message."""
    return (
        "Here is the raw product parameter data in JSON format:\n"
        f"{product_json}\n\n"
        "Generate the GEO-compliant English product detail page following every "
        "rule in the system prompt. Remember: copy every numeric rating exactly."
    )
