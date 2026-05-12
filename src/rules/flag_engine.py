from __future__ import annotations

from src.models import LabMeasurement

NUTRITION_TEST_MAP = {
    "vitamin d": "Low Vitamin D can be linked to low sun exposure or dietary insufficiency.",
    "vitamin b12": "Low Vitamin B12 can be associated with dietary or absorption issues.",
    "ferritin": "Low ferritin may suggest low iron stores.",
    "hemoglobin": "Low hemoglobin can suggest anemia and may need further workup.",
    "folate": "Low folate can be associated with nutritional deficiency.",
}

URGENT_KEYWORDS = {
    "troponin",
    "creatinine",
    "egfr",
    "potassium",
    "sodium",
    "wbc",
    "platelet",
    "glucose",
}

GENERIC_TEST_NAMES = {
    "rate",
    "result",
    "results",
    "value",
    "values",
    "range",
    "reference",
    "index",
    "ratio",
    "count",
    "level",
    "status",
    "comment",
    "remarks",
}


def build_rule_flags(
    measurements: list[LabMeasurement],
) -> tuple[list[str], list[str], list[str]]:
    concerns: list[str] = []
    doctor_signals: list[str] = []
    nutrition_signals: list[str] = []

    for item in measurements:
        if _is_generic_name(item.test_name):
            continue
        if item.status in {"low", "high"}:
            concerns.append(
                f"{item.test_name}: {item.value} {item.unit} (reference {item.reference_text}, {item.status})."
            )
            if _needs_doctor_followup(item):
                doctor_signals.append(
                    f"Discuss {item.test_name} with a doctor soon because it is outside the reference range."
                )

        nutrition_hint = _nutrition_signal(item)
        if nutrition_hint:
            nutrition_signals.append(nutrition_hint)

    return _dedupe(concerns), _dedupe(doctor_signals), _dedupe(nutrition_signals)


def _needs_doctor_followup(item: LabMeasurement) -> bool:
    name = item.test_name.lower()
    if item.severity >= 0.25:
        return True
    return any(keyword in name for keyword in URGENT_KEYWORDS) and item.status != "normal"


def _nutrition_signal(item: LabMeasurement) -> str | None:
    name = item.test_name.lower()
    if item.status != "low":
        return None
    for test_key, message in NUTRITION_TEST_MAP.items():
        if test_key in name:
            return f"{item.test_name}: {message}"
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _is_generic_name(name: str) -> bool:
    lowered = name.lower().strip()
    return lowered in GENERIC_TEST_NAMES
