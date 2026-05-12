from __future__ import annotations

import json
import re
import warnings
from typing import Any, TypedDict

from langchain_core.tools import tool

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change.*",
)

from langgraph.graph import END, StateGraph

from src.agents.prompts import (
    FLAG_TOOL_PROMPT_TEMPLATE,
    RETRIEVAL_TOOL_PROMPT_TEMPLATE,
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
    tool_trace: list[dict[str, Any]]
    output: AnalysisOutput
    error: str


def build_analysis_graph(llm, vector_store: LocalVectorStore):
    graph = StateGraph(WorkflowState)

    def parse_labs(state: WorkflowState) -> WorkflowState:
        text = state.get("report_text", "")
        measurements = extract_measurements(text)
        return {"measurements": measurements}

    def gemma_tool_call_node(state: WorkflowState) -> WorkflowState:
        measurements = state.get("measurements", [])
        if not measurements:
            raise RuntimeError(
                "No structured lab measurements were extracted. Mandatory Gemma "
                "tool calling cannot continue without measurements."
            )
        if not hasattr(llm, "bind_tools"):
            raise RuntimeError(
                "The configured LLM client does not expose bind_tools(). Gemma "
                "tool calling is required for this project."
            )

        @tool
        def flag_abnormal_results(measurements_json: str) -> str:
            """Flag out-of-range lab measurements and nutrition-related signals."""
            parsed_measurements = _measurements_from_json(measurements_json)
            concerns, doctor_signals, nutrition_signals = build_rule_flags(
                parsed_measurements
            )
            return json.dumps(
                {
                    "concerns": concerns,
                    "doctor_signals": doctor_signals,
                    "nutrition_signals": nutrition_signals,
                }
            )

        @tool
        def retrieve_medical_context(query: str) -> str:
            """Retrieve local report and medical knowledge context from ChromaDB."""
            docs = vector_store.retrieve_context(
                query=query, report_id=state["report_id"], k=4
            )
            return json.dumps(
                {
                    "context": "\n\n".join(doc.page_content for doc in docs),
                    "citations": [doc.metadata.get("source", "knowledge") for doc in docs],
                }
            )

        compact_measurements = json.dumps(
            [m.model_dump() for m in measurements], indent=2
        )

        flag_prompt = FLAG_TOOL_PROMPT_TEMPLATE.format(
            lab_measurements=compact_measurements
        )
        flag_calls = _invoke_required_tool_call(
            llm=llm,
            tools=[flag_abnormal_results],
            prompt=flag_prompt,
            required_tool_name="flag_abnormal_results",
        )

        updates: WorkflowState = {"tool_trace": []}
        flag_result = _execute_required_tool_call(
            call=flag_calls[0],
            tool_item=flag_abnormal_results,
            required_tool_name="flag_abnormal_results",
            default_args={"measurements_json": compact_measurements},
            updates=updates,
        )

        retrieval_prompt = RETRIEVAL_TOOL_PROMPT_TEMPLATE.format(
            lab_measurements=compact_measurements,
            flag_output=flag_result,
        )
        retrieval_calls = _invoke_required_tool_call(
            llm=llm,
            tools=[retrieve_medical_context],
            prompt=retrieval_prompt,
            required_tool_name="retrieve_medical_context",
        )
        _execute_required_tool_call(
            call=retrieval_calls[0],
            tool_item=retrieve_medical_context,
            required_tool_name="retrieve_medical_context",
            default_args={"query": _default_retrieval_query(measurements)},
            updates=updates,
        )

        return updates

    def llm_node(state: WorkflowState) -> WorkflowState:
        measurements = state.get("measurements", [])
        concerns = state.get("concerns", [])
        doctor_signals = state.get("doctor_signals", [])
        nutrition_signals = state.get("nutrition_signals", [])
        context = state.get("retrieved_context", "")

        compact_measurements = [m.model_dump() for m in measurements]
        user_prompt = USER_PROMPT_TEMPLATE.format(
            lab_measurements=json.dumps(compact_measurements, indent=2),
            concerns=json.dumps(concerns, indent=2),
            doctor_signals=json.dumps(doctor_signals, indent=2),
            nutrition_signals=json.dumps(nutrition_signals, indent=2),
            retrieved_context=context[:9000],
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
    graph.add_node("gemma_tool_call_node", gemma_tool_call_node)
    graph.add_node("llm_node", llm_node)
    graph.set_entry_point("parse_labs")
    graph.add_edge("parse_labs", "gemma_tool_call_node")
    graph.add_edge("gemma_tool_call_node", "llm_node")
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


def _invoke_required_tool_call(
    llm: Any, tools: list[Any], prompt: str, required_tool_name: str
) -> list[dict[str, Any]]:
    response = llm.bind_tools(tools).invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    tool_calls = getattr(response, "tool_calls", None) or []
    matching_calls = [
        call for call in tool_calls if call.get("name") == required_tool_name
    ]
    if not matching_calls:
        returned = [call.get("name", "<unnamed>") for call in tool_calls]
        raise RuntimeError(
            f"Gemma did not call required tool: {required_tool_name}. "
            f"Returned tool calls: {returned or 'none'}. This project requires "
            "Gemma function calling for analysis."
        )
    return matching_calls


def _execute_required_tool_call(
    call: dict[str, Any],
    tool_item: Any,
    required_tool_name: str,
    default_args: dict[str, Any],
    updates: WorkflowState,
) -> str:
    args = _tool_call_args(call)
    for key, value in default_args.items():
        if not args.get(key):
            args[key] = value

    result = tool_item.invoke(args)
    _apply_tool_result(updates, required_tool_name, result)
    updates["tool_trace"].append(
        {
            "name": required_tool_name,
            "status": "executed",
            "args": args,
            "result_preview": result[:500],
        }
    )
    return result


def _measurements_from_json(measurements_json: str) -> list[LabMeasurement]:
    try:
        raw_measurements = json.loads(measurements_json)
    except json.JSONDecodeError:
        return []

    measurements: list[LabMeasurement] = []
    for raw in raw_measurements:
        try:
            measurements.append(LabMeasurement.model_validate(raw))
        except Exception:
            continue
    return measurements


def _tool_call_args(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("args", {})
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _apply_tool_result(
    updates: WorkflowState, tool_name: str, result: str
) -> None:
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return

    if tool_name == "flag_abnormal_results":
        updates["concerns"] = parsed.get("concerns", [])
        updates["doctor_signals"] = parsed.get("doctor_signals", [])
        updates["nutrition_signals"] = parsed.get("nutrition_signals", [])
    elif tool_name == "retrieve_medical_context":
        updates["retrieved_context"] = parsed.get("context", "")
        updates["citations"] = parsed.get("citations", [])


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
