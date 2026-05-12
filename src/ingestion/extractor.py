from __future__ import annotations

import re

from src.models import LabMeasurement

LINE_PATTERN = re.compile(
    r"(?P<test>[A-Za-z][A-Za-z0-9()/%\-\s]{2,})\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[A-Za-z/%µu\.]+)?\s+"
    r"(?P<range>(?:\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?|<\s*\d+(?:\.\d+)?|>\s*\d+(?:\.\d+)?))"
)

UNIT_LIKE_TOKENS = {
    "iu/ml",
    "mg/dl",
    "mmol/l",
    "ng/ml",
    "pg/ml",
    "g/dl",
    "fl",
    "ml/min",
    "u/l",
    "meq/l",
    "%",
}

GENERIC_LABEL_TOKENS = {
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


def extract_measurements(text: str) -> list[LabMeasurement]:
    rows: list[LabMeasurement] = []
    seen: set[tuple[str, float, str]] = set()

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if len(line) < 8:
            continue

        match = LINE_PATTERN.search(line)
        if not match:
            continue

        test_name = _normalize_test_name(match.group("test"))
        if not _is_valid_test_name(test_name):
            continue
        value = float(match.group("value"))
        unit = (match.group("unit") or "").strip()
        reference_text = match.group("range").replace(" ", "")
        ref_low, ref_high = _parse_reference_range(reference_text)
        status, severity = _status_and_severity(value, ref_low, ref_high)

        key = (test_name.lower(), value, reference_text)
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            LabMeasurement(
                test_name=test_name,
                value=value,
                unit=unit,
                reference_text=reference_text,
                ref_low=ref_low,
                ref_high=ref_high,
                status=status,
                severity=severity,
            )
        )

    return rows


def _parse_reference_range(reference_text: str) -> tuple[float | None, float | None]:
    if "-" in reference_text:
        left, right = reference_text.split("-", maxsplit=1)
        return float(left), float(right)
    if reference_text.startswith("<"):
        return None, float(reference_text[1:])
    if reference_text.startswith(">"):
        return float(reference_text[1:]), None
    return None, None


def _status_and_severity(
    value: float, ref_low: float | None, ref_high: float | None
) -> tuple[str, float]:
    if ref_low is not None and value < ref_low:
        delta = (ref_low - value) / max(abs(ref_low), 1e-6)
        return "low", round(delta, 3)
    if ref_high is not None and value > ref_high:
        delta = (value - ref_high) / max(abs(ref_high), 1e-6)
        return "high", round(delta, 3)
    if ref_low is not None or ref_high is not None:
        return "normal", 0.0
    return "unknown", 0.0


def _normalize_test_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip()
    cleaned = cleaned.rstrip(":.-")
    # OCR often appends isolated digits to test names; remove trailing standalone numbers.
    cleaned = re.sub(r"\s+\d+(?:\.\d+)?$", "", cleaned).strip()
    return cleaned


def _is_valid_test_name(name: str) -> bool:
    lowered = name.lower().strip()
    if len(lowered) < 3:
        return False
    if lowered in UNIT_LIKE_TOKENS:
        return False

    letters = re.findall(r"[a-z]+", lowered)
    if not letters:
        return False
    if lowered in GENERIC_LABEL_TOKENS:
        return False
    if len(letters) == 1 and letters[0] in GENERIC_LABEL_TOKENS:
        return False

    # Reject names that look like unit noise (e.g., "iu/ml 0", "mg/dl").
    compact = lowered.replace(" ", "")
    if compact in UNIT_LIKE_TOKENS:
        return False
    if re.fullmatch(r"[a-z%/\.]+\d*", compact) and "/" in compact:
        return False

    return True
