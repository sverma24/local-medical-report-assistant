from __future__ import annotations

import json
import re
import warnings
from typing import Any, TypedDict

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change.*",
)

from langgraph.graph import END, StateGraph

from src.agents.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from src.ingestion.extractor import extract_measurements
from src.models import AnalysisOutput, LabMeasurement
from src.retrieval.vector_store import LocalVectorStore
from src.rules.flag_engine import build_rule_flags


class WorkflowState(TypedDict, total=False):
    report_id: str
    report_text: str
    measurements: list[LabMeasurement]
    concerns: list[str]
    doctor_signals: list[str]
    nutrition_signals: list[str]
    retrieved_context: str
    citations: list[str]
    output: AnalysisOutput
    error: str


def build_analysis_graph(llm, vector_store: LocalVectorStore):
    graph = StateGraph(WorkflowState)

    def parse_labs(state: WorkflowState) -> WorkflowState:
        text = state.get("report_text", "")
        measurements = extract_measurements(text)
        return {"measurements": measurements}

    def deterministic_prep_node(state: WorkflowState) -> WorkflowState:
        measurements = state.get("measurements", [])
        if not measurements:
            raise RuntimeError(
                "No structured lab measurements were extracted. Analysis cannot continue."
            )

        concerns, doctor_signals, nutrition_signals = build_rule_flags(measurements)
        query = _default_retrieval_query(measurements)
        docs = vector_store.retrieve_context(query=query, report_id=state["report_id"])
        return {
            "concerns": concerns,
            "doctor_signals": doctor_signals,
            "nutrition_signals": nutrition_signals,
            "retrieved_context": "\n\n".join(doc.page_content for doc in docs),
            "citations": [doc.metadata.get("source", "knowledge") for doc in docs],
        }

    def llm_node(state: WorkflowState) -> WorkflowState:
        measurements = state.get("measurements", [])
        concerns = state.get("concerns", [])
        doctor_signals = state.get("doctor_signals", [])
        nutrition_signals = state.get("nutrition_signals", [])
        context = state.get("retrieved_context", "")
        llm_measurements = _prepare_measurements_for_llm(
            measurements, vector_store.config.max_measurements_for_llm
        )
        user_prompt = USER_PROMPT_TEMPLATE.format(
            lab_measurements=json.dumps(llm_measurements, indent=2),
            concerns=json.dumps(concerns, indent=2),
            doctor_signals=json.dumps(doctor_signals, indent=2),
            nutrition_signals=json.dumps(nutrition_signals, indent=2),
            retrieved_context=context[: vector_store.config.retrieved_context_char_limit],
        )
        response = llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        parsed = _parse_llm_json(response.content)

        merged = AnalysisOutput(
            summary=parsed.summary,
            areas_of_concern=_merge_concern_lists(
                parsed.areas_of_concern, concerns, measurements
            ),
            doctor_followup=_merge_concern_lists(
                parsed.doctor_followup, doctor_signals, measurements
            ),
            nutritional_signals=_merge_lists(parsed.nutritional_signals, nutrition_signals),
            lifestyle_recommendations=parsed.lifestyle_recommendations,
            questions_for_doctor=parsed.questions_for_doctor,
            disclaimer=parsed.disclaimer
            or "Informational only. This does not replace medical advice from a licensed clinician.",
        )
        return {"output": merged}

    graph.add_node("parse_labs", parse_labs)
    graph.add_node("deterministic_prep_node", deterministic_prep_node)
    graph.add_node("llm_node", llm_node)
    graph.set_entry_point("parse_labs")
    graph.add_edge("parse_labs", "deterministic_prep_node")
    graph.add_edge("deterministic_prep_node", "llm_node")
    graph.add_edge("llm_node", END)
    return graph.compile()


def _parse_llm_json(content: Any) -> AnalysisOutput:
    if not isinstance(content, str):
        return AnalysisOutput()

    candidate = content.strip()
    if "```" in candidate:
        candidate = candidate.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(candidate)
        return AnalysisOutput.model_validate(data)
    except Exception:
        return AnalysisOutput(
            summary="Could not fully structure the model output. Please review extracted concerns manually.",
            disclaimer="Informational only. This does not replace medical advice from a licensed clinician.",
        )


def _merge_lists(first: list[str], second: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in first + second:
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _prepare_measurements_for_llm(
    measurements: list[LabMeasurement], limit: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    abnormal = [m for m in measurements if m.status in {"high", "low"}]
    normal = [m for m in measurements if m.status not in {"high", "low"}]
    selected = (abnormal + normal)[:limit]

    return [
        {
            "test_name": m.test_name,
            "value": m.value,
            "unit": m.unit,
            "reference_text": m.reference_text,
            "status": m.status,
            "severity": m.severity,
        }
        for m in selected
    ]


def _default_retrieval_query(measurements: list[LabMeasurement]) -> str:
    abnormal_tests = [m.test_name for m in measurements if m.status in {"low", "high"}]
    if abnormal_tests:
        return " ".join(abnormal_tests)
    return "medical report interpretation and lifestyle guidance"


def _merge_concern_lists(
    first: list[str], second: list[str], measurements: list[LabMeasurement]
) -> list[str]:
    seen_keys: set[str] = set()
    seen_normalized: set[str] = set()
    result: list[str] = []
    measurement_names = [m.test_name for m in measurements]

    for item in first + second:
        value = item.strip()
        if not value:
            continue

        key = _extract_measurement_key(value, measurement_names)
        normalized = _normalize_for_dedupe(value)

        if key and key in seen_keys:
            continue
        if normalized in seen_normalized:
            continue

        if key:
            seen_keys.add(key)
        seen_normalized.add(normalized)
        result.append(value)

    return result


def _extract_measurement_key(line: str, measurement_names: list[str]) -> str | None:
    normalized_line = _normalize_for_dedupe(line)
    for name in measurement_names:
        normalized_name = _normalize_for_dedupe(name)
        if not normalized_name:
            continue
        if normalized_name in normalized_line:
            return normalized_name
    return None


def _normalize_for_dedupe(value: str) -> str:
    cleaned = value.lower()
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
